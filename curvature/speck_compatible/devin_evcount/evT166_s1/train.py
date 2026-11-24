"""Speck-fitting 9-core trainer — drop-in replacement for train_10ms_4striding_elevate.py.

Backbone swapped to a topology VERIFIED against the real core-mapping solver
(sinabs map_layers_to_cores -> {1:0, 2:1, 3:2, 4:8, 5:4, 6:3, 7:6, 8:5, 9:7}).
Shape mirrors a proven deployed SynSense SNN topology: downsampling by strided convs,
channel taper 16->32->32->32->32->64->64->64, AvgPool->Linear readout to 2 class features.

IMPORTANT: the previous architecture could NOT map (layers exceeded per-core
kernel/neuron memory). check_compatibility.py only instantiates DynapcnnNetwork,
which does NOT run the mapper — always verify with DynapcnnNetwork.memory_summary()
+ make_config() before training.

Dataset contract identical to the elevate trainer:
    .npz files with keys 'frames' (T, 2, 128, 128) and 'label' in {0, 1}
    forward() strides time ::4 and returns mean spike-rate logits (B, 2).
"""

import matplotlib
matplotlib.use('Agg')  # headless on HPC
import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, classification_report
import sinabs
import sinabs.layers as sl
import matplotlib.pyplot as plt
import seaborn as sns
import gc

gc.collect()
torch.cuda.empty_cache()

# --- 1. CONFIGURATION ---
DATA_PATH = "/cfs/earth/scratch/matheron/tactile-sensing/data/count_bins/dataset_curv_evT166"
BATCH_SIZE = 10
NUM_WORKERS = 8
EPOCHS = 100
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Optional warm-start: leave empty to train from scratch
PRETRAINED_SOURCE_PATH = ""
AUG_FLIP = False
AUG_SHIFT = False
AUG_DROP = True
AUG_STRETCH = True


# --- 2. DATASET (identical to elevate trainer) ---
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


# --- 3. HARDWARE-VERIFIED 9-CORE TOPOLOGY ---
def _iaf(target_batch_size):
    return sl.IAFSqueeze(batch_size=target_batch_size, spike_threshold=1.0,
                         min_v_mem=-1.0,
                         surrogate_grad_fn=sinabs.activation.PeriodicExponential())


class SpeckFittingTrainer(nn.Module):
    """9 DynapcnnLayers: 8 conv blocks + Linear readout.

    Channel taper = MAX verified by demo/maximize.py sweep:
    (16, 64, 32, 32, 32, 64, 64, 64) — 6/9 cores at 100% kernel memory.
    Per-layer fit vs per-core limits (kernel/neuron KiB):
        core0-2: 16/64 | core3-4: 32/32 | core5-6: 64/16 | core7-8: 16/16
    """

    def __init__(self, target_batch_size):
        super().__init__()
        self.spiking_backbone = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1, bias=False),   # 128->64  n:64Ki k:0.5Ki -> c0-2
            _iaf(target_batch_size),
            nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64->32   n:64Ki k:16Ki -> c1 (MAX)
            _iaf(target_batch_size),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 32->16   n:8Ki  k:32Ki -> c3/4
            _iaf(target_batch_size),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 16->16   n:8Ki  k:16Ki
            _iaf(target_batch_size),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 16->8    n:2Ki  k:16Ki
            _iaf(target_batch_size),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 8->4     n:1Ki  k:32Ki -> c3-6
            _iaf(target_batch_size),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 4->4     n:1Ki  k:64Ki -> c5-6
            _iaf(target_batch_size),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 4->4     n:1Ki  k:64Ki -> c5-6
            _iaf(target_batch_size),
            nn.AvgPool2d(2, 2),                                                # ->2x2    (sum-pool on chip)
            nn.Flatten(),
            nn.Linear(256, 2, bias=False),                                     # 2 class features -> c7/8
            _iaf(target_batch_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.shape
        x = x[:, ::1]                       # time-stride every 4th bin (same as elevate)
        new_timesteps = x.shape[1]
        x = x.reshape(batch_size * new_timesteps, C, H, W)
        out = self.spiking_backbone(x)      # Linear+IAF outputs (B*T', 2)
        out = out.reshape(batch_size, new_timesteps, -1)
        return out.mean(dim=1)              # mean spike-rate logits -> CE loss


model = SpeckFittingTrainer(target_batch_size=BATCH_SIZE).to(DEVICE)

# Mandatory warm-up: allocates IAFSqueeze v_mem buffers before anything else
print("Warming up Sinabs layer dimensions...", flush=True)
dummy_input = torch.zeros((BATCH_SIZE, 40, 2, 128, 128), device=DEVICE)
with torch.no_grad():
    _ = model(dummy_input)

if PRETRAINED_SOURCE_PATH and os.path.exists(PRETRAINED_SOURCE_PATH):
    print(f"Warm-starting from: {PRETRAINED_SOURCE_PATH}")
    sd = torch.load(PRETRAINED_SOURCE_PATH, map_location=DEVICE)
    model.load_state_dict({k: v for k, v in sd.items() if not k.endswith(".v_mem")}, strict=False)
else:
    print("Training from scratch (no warm-start checkpoint).")

# --- 4. SAFETY: verify the backbone really maps BEFORE burning GPU hours ---
# NOTE: DynapcnnNetwork's graph extractor probes the backbone with a CPU sample
# tensor — run the check on a CPU deepcopy, leave the training model on DEVICE.
print("Verifying layer->core mapping (the check that was missing before)...", flush=True)
import copy
from sinabs.backend.dynapcnn import DynapcnnNetwork
_snn_cpu = copy.deepcopy(model.spiking_backbone).cpu()
_hw = DynapcnnNetwork(snn=_snn_cpu, discretize=True,
                      input_shape=(2, 128, 128), dvs_input=True)
ms = _hw.memory_summary()
for i in sorted(ms['kernel']):
    print(f"   layer {i}: kernel={ms['kernel'][i]/1024:.1f}Ki  neuron={ms['neuron'][i]/1024:.1f}Ki")
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
_builder = ChipFactory("speck2fdevkit:0").get_config_builder()
print("   mapping:", _builder.map_layers_to_cores(_hw.dynapcnn_layers), flush=True)
del _hw, _snn_cpu

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))


def save_plots(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Flat', 'Curved'], yticklabels=['Flat', 'Curved'])
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f'{name.lower().replace(" ", "_")}_matrix.png')
    plt.close()


# --- 5. TRAINING & VALIDATION LOOP (identical discipline to elevate) ---
best_val_acc = 0.0
best_model_path = "best_speck_fitting.pt"
total_train_batches = len(train_loader)
global_start_time = time.time()
last_heartbeat_time = time.time()

print("TRAINING START: fitting 9-core SNN", flush=True)
for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        sinabs.utils.reset_states(model.spiking_backbone)

        with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # weight discipline: keep |W| < spike_threshold=1.0 (chip fires once per crossing)
        with torch.no_grad():
            for m in model.modules():
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
