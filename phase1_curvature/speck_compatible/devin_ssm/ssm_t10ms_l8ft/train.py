"""Speck-SNN front-end + SSM readout head — flat/curved sequence classifier.

The 9-layer Speck-fitting SNN (MAX taper, verified core mapping) is kept as a
FROZEN front-end feature extractor, warm-started from a trained
checkpoint; only the SSM head is trained end-to-end with a small diagonal state-space readout
head (S4D-style, pure PyTorch, no mamba dep). Instead of the mean-over-time
readout the chip uses, the per-timestep output of the Speck SNN's LAST CONV
layer — the (T', 64) spatially-pooled post-IAF feature spikes — feeds the
SSM, which integrates the whole episode and emits the class. The SNN is used
purely as a spike feature extractor; the class Linear is bypassed entirely.

NOT chip-deployable — this is the software-side "smart temporal readout" arm
of the chip-vs-software comparison protocol (see KB chip_sim_validation.md:
the SNN is a sequence classifier; the signal lives in contact dynamics, so a
state-space head is the natural readout for it). Backbone keeps the |W|<1
wclamp discipline so a deployable core model could still be recovered.

Variant: ssm_t10ms_l8ft — TIME-BINNED data only (dataset_curv_padded_10ms,
T=1331, ::4 -> 333 steps). FEAT_MODE=l8: (T',64) last-conv features, not the
(T',2) readout spikes. JOINT FINE-TUNE of the warm-started
backbone (lr=LR/10) + 0.3 aux readout CE. One dataset family per variant — no mixing.
"""

import matplotlib
matplotlib.use('Agg')  # headless on HPC
import math
import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import sinabs
import sinabs.layers as sl
import matplotlib.pyplot as plt
import seaborn as sns
import gc

gc.collect()
torch.cuda.empty_cache()

# --- 1. CONFIGURATION ---
DATA_PATH = "/cfs/earth/scratch/matheron/tactile-sensing/data/time_bins/dataset_curv_padded_10ms"
BACKBONE_CKPT = "/cfs/earth/scratch/matheron/tactile-sensing/deploy_checkpoints/wclamp_v93.33_t86.67.pt"
TIME_STRIDE = 4                    # 10ms bins: T=1331 -> 333 steps (same as its trainer)
FREEZE_BACKBONE = False            # warm-start + joint fine-tune (backbone lr=LR/10)
FEAT_MODE = "l8"                   # 'readout' (T',2) | 'logit' (pre-IAF) | 'l8' (T',64)
BATCH_SIZE = 10
NUM_WORKERS = 8
EPOCHS = 100
LEARNING_RATE = 1e-3               # SSM head lr (backbone's was 2e-4)
WEIGHT_DECAY = 1e-4
D_MODEL = 64
D_STATE = 64
N_SSM_LAYERS = 3
DROPOUT = 0.1
AUX_LOSS_W = 0.3                   # aux mean-rate CE on the bypassed readout —
                                   # keeps class signal in the backbone during ft
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUG_FLIP = False
AUG_SHIFT = False
AUG_DROP = True
AUG_STRETCH = True


# --- 2. DATASET (identical contract + augs to the speck-fitting trainer) ---
class PaddedSpeckTactileDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
        if len(self.file_list) == 0:
            raise FileNotFoundError(f"No .npz files detected in {data_dir}.")
        print(f"Dataset initialized with {len(self.file_list)} padded samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        with np.load(file_path) as loaded:
            data = torch.from_numpy(loaded['frames']).float()
            label = torch.tensor(int(loaded['label']), dtype=torch.long)

        if torch.is_grad_enabled():
            if np.random.rand() > 0.5:
                shift = np.random.randint(-12, 12)
                data = torch.roll(data, shifts=shift, dims=0)
            if AUG_FLIP:
                if np.random.rand() > 0.5: data = torch.flip(data, dims=[-1])
                if np.random.rand() > 0.5: data = torch.flip(data, dims=[-2])
            if AUG_SHIFT:
                dx = int(np.random.randint(-16, 17)); dy = int(np.random.randint(-16, 17))
                data = torch.roll(data, shifts=(dy, dx), dims=(-2, -1))
            if AUG_DROP:
                drop = (torch.rand_like(data) < 0.10).float()
                data = data * (1.0 - drop)
            if AUG_STRETCH:
                s = float(np.random.uniform(0.8, 1.2))
                t = data.shape[0]
                new_t = max(2, int(round(t / s)))
                idx = torch.linspace(0, t - 1, new_t).round().long().clamp(0, t - 1)
                d2 = data[idx]
                if new_t < t:   # faster motion -> zero-pad tail (matches dataset padding)
                    data = torch.cat([d2, torch.zeros((t - new_t,) + d2.shape[1:], dtype=d2.dtype)], dim=0)
                else:           # slower motion -> random crop back to t
                    off = int(np.random.randint(0, new_t - t + 1))
                    data = d2[off:off + t]

        data = torch.clamp(data, 0.0, 1.0)
        return data, label


full_dataset = PaddedSpeckTactileDataset(data_dir=DATA_PATH)
total_count = len(full_dataset)
train_size = int(0.70 * total_count)
val_size = int(0.15 * total_count)
test_size = total_count - train_size - val_size

train_subset, val_subset, test_subset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)
print(f"Splits -> Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_subset)}")

LOADER_ARGS = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS,
               'pin_memory': True, 'persistent_workers': True, 'drop_last': True}
train_loader = DataLoader(train_subset, shuffle=True, **LOADER_ARGS)
val_loader = DataLoader(val_subset, shuffle=False, **LOADER_ARGS)
test_loader = DataLoader(test_subset, shuffle=False, **LOADER_ARGS)


# --- 3a. SPECK 9-CORE BACKBONE (identical topology to the fitting trainer) ---
def _iaf(target_batch_size):
    return sl.IAFSqueeze(batch_size=target_batch_size, spike_threshold=1.0,
                         min_v_mem=-1.0,
                         surrogate_grad_fn=sinabs.activation.PeriodicExponential())


# --- 3b. SSM HEAD — diagonal state-space model (S4D-Lin), pure PyTorch ---
class S4DLayer(nn.Module):
    """One diagonal SSM layer:  h_t = exp(dt*A) h_{t-1} + B_bar u_t ;  y_t = Re(C h_t).

    Complex diagonal A per channel, S4D-Lin init (-0.5 + i*pi*k), zero-order-
    hold discretization. Integrates the readout stream's temporal structure —
    exactly what the mean-rate readout discards.
    """

    def __init__(self, d_model, d_state=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)                                       # (D,)
        self.A_re = nn.Parameter(-0.5 * torch.ones(d_model, d_state))
        self.A_im = nn.Parameter(math.pi * torch.arange(d_state).float()
                                 .unsqueeze(0).repeat(d_model, 1))               # (D,N)
        self.B = nn.Parameter(torch.randn(d_model, d_state, 2) / math.sqrt(2 * d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state, 2) / math.sqrt(d_state))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, u):                    # u: (B, T, D) float32
        dt = self.log_dt.exp()                                        # (D,)
        A = torch.complex(self.A_re, self.A_im)                       # (D,N)
        dA = A * dt[:, None]
        Abar = torch.exp(dA)                                          # (D,N)
        Bbar = torch.view_as_complex(self.B.float()) * dt[:, None] \
            * (Abar - 1.0) / dA                                       # ZOH
        C = torch.view_as_complex(self.C.float())                     # (D,N)
        h = u.new_zeros(u.shape[0], u.shape[2], A.shape[1], dtype=torch.cfloat)
        ys = []
        uh = u.transpose(0, 1)                                        # (T,B,D)
        for t in range(u.shape[1]):
            h = Abar * h + uh[t][:, :, None] * Bbar
            ys.append(torch.einsum('bdn,dn->bd', h, C).real)
        return self.out(torch.stack(ys, dim=1))                       # (B,T,D)


class SSMBlock(nn.Module):
    """Pre-norm S4D + pre-norm GLU-ish FFN, residual around both."""

    def __init__(self, d_model, d_state, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.ssm = S4DLayer(d_model, d_state)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Linear(2 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.ssm(self.n1(x)))
        return x + self.drop(self.ff(self.n2(x)))


class SSMHead(nn.Module):
    """feat (B,T,F) -> proj -> N x SSMBlock -> norm -> episode mean-pool -> logits."""

    def __init__(self, feat_dim, d_model, d_state, n_layers, n_classes, dropout):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.blocks = nn.ModuleList(SSMBlock(d_model, d_state, dropout)
                                    for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, feat):
        h = self.proj(feat)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h).mean(dim=1))


class SpeckSSMClassifier(nn.Module):
    """9-core Speck SNN (frozen) + SSM readout over its per-timestep output."""

    def __init__(self, target_batch_size):
        super().__init__()
        self.spiking_backbone = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1, bias=False),
            _iaf(target_batch_size),
            nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1, bias=False),
            _iaf(target_batch_size),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            _iaf(target_batch_size),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            _iaf(target_batch_size),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            _iaf(target_batch_size),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            _iaf(target_batch_size),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            _iaf(target_batch_size),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            _iaf(target_batch_size),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256, 2, bias=False),
            _iaf(target_batch_size),
        )
        feat_dim = {"readout": 2, "logit": 2, "l8": 64}[FEAT_MODE]
        self.ssm_head = SSMHead(feat_dim, D_MODEL, D_STATE, N_SSM_LAYERS, 2, DROPOUT)
        for m in self.spiking_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _features(self, x):                  # x: (B*T', 2, 128, 128)
        if FEAT_MODE == "readout":
            return self.spiking_backbone(x)              # post-IAF spikes (B*T',2)
        if FEAT_MODE == "logit":
            return self.spiking_backbone[:-1](x)         # pre-IAF readout logits
        f = self.spiking_backbone[:-4](x)                # last conv+IAF (B*T',64,4,4)
        if AUX_LOSS_W:
            self._ro = self.spiking_backbone[-4:](f)     # bypassed readout for aux CE
        return f.mean(dim=(2, 3))                        # spatial pool -> 64

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.shape
        x = x[:, ::TIME_STRIDE]
        new_timesteps = x.shape[1]
        x = x.reshape(batch_size * new_timesteps, C, H, W)
        if FREEZE_BACKBONE:
            with torch.no_grad():
                feat = self._features(x)
        else:
            feat = self._features(x)
        feat = feat.reshape(batch_size, new_timesteps, -1)
        self.last_rate = feat.mean(dim=1) if feat.shape[-1] == 2 else None
        if AUX_LOSS_W and getattr(self, "_ro", None) is not None:
            self.last_rate = self._ro.reshape(batch_size, new_timesteps, -1).mean(dim=1)
            self._ro = None
        return self.ssm_head(feat)


model = SpeckSSMClassifier(target_batch_size=BATCH_SIZE).to(DEVICE)

# Mandatory warm-up: allocates IAFSqueeze v_mem buffers before weight load
print("Warming up Sinabs layer dimensions...", flush=True)
dummy_input = torch.zeros((BATCH_SIZE, 40, 2, 128, 128), device=DEVICE)
with torch.no_grad():
    _ = model(dummy_input)

if BACKBONE_CKPT and os.path.exists(BACKBONE_CKPT):
    print(f"Warm-starting backbone from: {BACKBONE_CKPT}", flush=True)
    sd = torch.load(BACKBONE_CKPT, map_location=DEVICE)
    n_missing = model.load_state_dict(
        {k: v for k, v in sd.items() if not k.endswith(".v_mem")}, strict=False)
    print(f"   backbone weights loaded (unloaded keys = SSM head, fresh)", flush=True)
else:
    print("Training backbone from scratch too (no warm-start checkpoint).", flush=True)

if FREEZE_BACKBONE:
    for p in model.spiking_backbone.parameters():
        p.requires_grad = False
    trainable = model.ssm_head.parameters()
    n_tr = sum(p.numel() for p in trainable)
    print(f"Backbone FROZEN — training SSM head only ({n_tr} params)", flush=True)
    optimizer = torch.optim.AdamW(model.ssm_head.parameters(),
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    print("Fine-tuning backbone jointly (head lr=LR, backbone lr=LR/10)", flush=True)
    optimizer = torch.optim.AdamW(
        [{'params': model.ssm_head.parameters(), 'lr': LEARNING_RATE},
         {'params': model.spiking_backbone.parameters(), 'lr': LEARNING_RATE * 0.1}],
        weight_decay=WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


def save_plots(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Flat', 'Curved'], yticklabels=['Flat', 'Curved'])
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f'{name.lower().replace(" ", "_")}_matrix.png')
    plt.close()


# --- 5. TRAINING & VALIDATION LOOP (same discipline as the speck trainer) ---
best_val_acc = 0.0
best_model_path = "best_ssm.pt"
total_train_batches = len(train_loader)
global_start_time = time.time()
last_heartbeat_time = time.time()

print(f"TRAINING START: Speck-SNN({FEAT_MODE}) + SSM head "
      f"(d{D_MODEL}/s{D_STATE}x{N_SSM_LAYERS})", flush=True)
for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        sinabs.utils.reset_states(model.spiking_backbone)

        output = model(data)
        loss = criterion(output, target)
        if AUX_LOSS_W and model.last_rate is not None:
            loss = loss + AUX_LOSS_W * criterion(model.last_rate, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()

        # chip weight discipline applies to the backbone only (SSM is free)
        if not FREEZE_BACKBONE:
            with torch.no_grad():
                for m in model.spiking_backbone.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        m.weight.clamp_(-1.0, 1.0)

        batch_loss_val = loss.item()
        if np.isnan(batch_loss_val) or np.isinf(batch_loss_val):
            print(f"NaN loss at batch {i+1}, aborting run.", flush=True)
            break
        total_loss += batch_loss_val

        current_time = time.time()
        if current_time - last_heartbeat_time >= 120:
            elapsed = current_time - global_start_time
            progress = (i + 1) / total_train_batches * 100
            print(f"   [TRAIN] Epoch {epoch+1:02d} | {progress:.1f}% | loss {batch_loss_val:.4f} | {elapsed/60:.1f}m", flush=True)
            last_heartbeat_time = current_time

    avg_train_loss = total_loss / total_train_batches
    if np.isnan(avg_train_loss) or np.isinf(avg_train_loss):
        break

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            sinabs.utils.reset_states(model.spiking_backbone)
            output = model(data)
            val_preds.extend(output.argmax(1).cpu().numpy())
            val_targets.extend(target.cpu().numpy())

    val_acc = 100. * np.sum(np.array(val_preds) == np.array(val_targets)) / len(val_targets)

    is_best = False
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        is_best = True
        save_plots(val_targets, val_preds, f"Epoch_{epoch+1}")

    marker = "  [SAVED NEW BEST]" if is_best else ""
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss {avg_train_loss:.4f} | val {val_acc:.2f}% | best {best_val_acc:.2f}% | {time.time()-epoch_start_time:.1f}s{marker}", flush=True)
    scheduler.step()

# --- 6. FINAL TEST ---
print("\nLoading best checkpoint for final test...", flush=True)
if os.path.exists(best_model_path):
    sd = torch.load(best_model_path)
    model.load_state_dict({k: v for k, v in sd.items() if not k.endswith(".v_mem")}, strict=False)
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        sinabs.utils.reset_states(model.spiking_backbone)
        test_preds.extend(model(data).argmax(1).cpu().numpy())
        test_targets.extend(target.cpu().numpy())

test_acc = 100. * np.sum(np.array(test_preds) == np.array(test_targets)) / len(test_targets)
print(f"\nFINAL UNBIASED TEST ACCURACY: {test_acc:.2f}%")
save_plots(test_targets, test_preds, "Final Test Validation")
print(classification_report(test_targets, test_preds,
                            target_names=['Flat', 'Curved'], zero_division=0), flush=True)
