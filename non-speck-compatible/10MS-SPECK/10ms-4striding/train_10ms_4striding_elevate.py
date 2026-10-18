import matplotlib
matplotlib.use('Agg') # Enforce headless background rendering for HPC clusters
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
import sinabs.activation as sa
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Force Python garbage collection and clear stale CUDA allocation pools
gc.collect()
torch.cuda.empty_cache()


# --- 1. CONFIGURATION (HARDWARE ALIGNED) ---
DATA_PATH = "/cfs/earth/scratch/matheron/tactile-sensing/data/dataset_curv_padded_10ms" 
BATCH_SIZE = 10         
NUM_WORKERS = 8         
EPOCHS = 100
LEARNING_RATE = 2e-4    
WEIGHT_DECAY = 1e-4  # 🎯 ELEVATION: Regularizes synaptic weight shifts to prevent loss stagnation
DEVICE = torch.device("cuda")

# --- 2. HARDWARE-SAFE DATASET ---
class PaddedSpeckTactileDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
        
        if len(self.file_list) == 0:
            raise FileNotFoundError(f"⚠️ No .npz files detected in {data_dir}. Verify preprocessing completed.")
        print(f"📦 Dataset initialized with {len(self.file_list)} fully uniform, padded samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        with np.load(file_path) as loaded:
            data = torch.from_numpy(loaded['frames']).float()
            label = torch.tensor(int(loaded['label']), dtype=torch.long)
        
        # 🎯 ELEVATION: Inject small random temporal roll/jitter augmentation during training mode
        # This breaks up memorized temporal frame patterns on your 200 pilot samples
        if torch.is_grad_enabled() and np.random.rand() > 0.5:
            shift = np.random.randint(-12, 12)
            data = torch.roll(data, shifts=shift, dims=0)

        # Clamp inputs to binary or low integer space to prevent neuron membrane explosion
        data = torch.clamp(data, 0.0, 1.0)
        return data, label

# Instantiate full dataset
full_dataset = PaddedSpeckTactileDataset(data_dir=DATA_PATH)

# Perform clean random splits (70% Train, 15% Validation, 15% Test)
total_count = len(full_dataset)
train_size = int(0.70 * total_count)
val_size = int(0.15 * total_count)
test_size = total_count - train_size - val_size

train_subset, val_subset, test_subset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)

print(f"📊 Dataset splits mapped cleanly -> Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_subset)}")

# DataLoaders configuration (drop_last=True for Train to keep batch shapes uniform)
LOADER_ARGS_TRAIN = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'pin_memory': True, 'persistent_workers': True, 'drop_last': True}
LOADER_ARGS_EVAL  = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'pin_memory': True, 'persistent_workers': True, 'drop_last': True}

train_loader = DataLoader(train_subset, shuffle=True,  **LOADER_ARGS_TRAIN)
val_loader   = DataLoader(val_subset,   shuffle=False, **LOADER_ARGS_EVAL)
test_loader  = DataLoader(test_subset,  shuffle=False, **LOADER_ARGS_EVAL)

# --- 3. HARDWARE-VERIFIED SILICON TOPO ---
class Speck9LayerTrainer(nn.Module):
    def __init__(self, target_batch_size):
        super().__init__()
        import sinabs.activation
        
        self.spiking_backbone = nn.Sequential(
            # Core 0: Downsample step
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False), 
            sl.IAFSqueeze(batch_size=target_batch_size, spike_threshold=1.0, min_v_mem=-1.0, surrogate_grad_fn=sinabs.activation.PeriodicExponential()), 

            # Core 1:
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False), 
            sl.IAFSqueeze(batch_size=target_batch_size, spike_threshold=1.0, min_v_mem=-1.0, surrogate_grad_fn=sinabs.activation.PeriodicExponential()), 
            sl.SumPool2d(kernel_size=2, stride=2),

            # Core 2: Bottleneck
            nn.Conv2d(48, 64, kernel_size=1, bias=False), 
            sl.IAFSqueeze(batch_size=target_batch_size, spike_threshold=1.0, min_v_mem=-1.0, surrogate_grad_fn=sinabs.activation.PeriodicExponential()), 
            sl.SumPool2d(kernel_size=2, stride=2),

            # Core 3:
            nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=False), 
            sl.IAFSqueeze(batch_size=target_batch_size, spike_threshold=1.0, min_v_mem=-1.0, surrogate_grad_fn=sinabs.activation.PeriodicExponential()),

            # Core 4:
            nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=False), 
            sl.IAFSqueeze(batch_size=target_batch_size, spike_threshold=1.0, min_v_mem=-1.0, surrogate_grad_fn=sinabs.activation.PeriodicExponential()), 
            sl.SumPool2d(kernel_size=2, stride=2),

            # Core 5:
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False), 
            sl.IAFSqueeze(batch_size=target_batch_size, spike_threshold=1.0, min_v_mem=-1.0, surrogate_grad_fn=sinabs.activation.PeriodicExponential()),

            # Core 6:
            nn.Conv2d(96, 128, kernel_size=2, padding=1, bias=False), 
            sl.IAFSqueeze(batch_size=target_batch_size, spike_threshold=1.0, min_v_mem=-1.0, surrogate_grad_fn=sinabs.activation.PeriodicExponential()), 
            sl.SumPool2d(kernel_size=2, stride=2),

            # Core 7: Bottleneck
            nn.Conv2d(128, 32, kernel_size=1, bias=False),
            sl.IAFSqueeze(batch_size=target_batch_size, spike_threshold=1.0, min_v_mem=-1.0, surrogate_grad_fn=sinabs.activation.PeriodicExponential()),

            # Core 8: Convolutional Readout Layer
            nn.Conv2d(32, 2, kernel_size=4, bias=False),
            sl.IAFSqueeze(batch_size=target_batch_size, spike_threshold=1.0, min_v_mem=-1.0, surrogate_grad_fn=sinabs.activation.PeriodicExponential()),
            nn.Flatten()
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.shape
        
        # ⚡ STRIDE THE TIME DIMENSION (Take every 4th frame)
        x = x[:, ::4, :, :, :] 
        new_timesteps = x.shape[1]
        
        x = x.reshape(batch_size * new_timesteps, C, H, W)
        out_spikes = self.spiking_backbone(x)
        out_spikes = out_spikes.reshape(batch_size, new_timesteps, -1)
        
        return out_spikes.mean(dim=1)


model = Speck9LayerTrainer(target_batch_size=BATCH_SIZE).to(DEVICE)

# 🎯 FIX: Warm up the model with a dummy tensor to shape Sinabs memory parameters!
# Expected shape: [Batch, Time, Channels, Height, Width]
# Based on your 4-striding configuration, any mock timeline length (like 40) works fine.
print("⚡ Warming up Sinabs layer dimensions...", flush=True)
dummy_input = torch.zeros((BATCH_SIZE, 40, 2, 128, 128), device=DEVICE)
with torch.no_grad():
    _ = model(dummy_input)
    
PRETRAINED_SOURCE_PATH = "/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/10MS-SPECK/10ms-4striding/4stride-80%/best_10ms_speck_compatible.pt"

if os.path.exists(PRETRAINED_SOURCE_PATH):
    print(f"🔄 Found pretrained weights at: {PRETRAINED_SOURCE_PATH}")
    print("🚀 Loading checkpoint as baseline starting point...", flush=True)
    model.load_state_dict(torch.load(PRETRAINED_SOURCE_PATH, map_location=DEVICE))
    print("✅ Baseline weights loaded successfully!")
else:
    raise FileNotFoundError(f"⚠️ Critical error: Target baseline checkpoint not found at {PRETRAINED_SOURCE_PATH}. Check your folder structure.")

# 🎯 ELEVATION: Switched to AdamW with weight decay to stabilize synaptic parameter drifting
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 🎯 ELEVATION: Added Label Smoothing to mitigate loss collapse down to zero
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 🎯 ELEVATION: Cosine Annealing scheduler smoothly decays steps to minimize late-stage volatility
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

scaler = GradScaler()

def save_plots(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Flat', 'Curved'], yticklabels=['Flat', 'Curved'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f'{name.lower().replace(" ", "_")}_matrix.png')
    plt.close()

# --- 4. TRAINING & VALIDATION LOOP ---
# 🎯 FIX: Match target_best threshold with checkpoint validation score to protect file structure
best_val_acc = 80.00
best_model_path = "best_10ms_speck_compatible.pt"
total_train_batches = len(train_loader)
global_start_time = time.time()
last_heartbeat_time = time.time()

print(f"🚀 TRAINING START: Silicon-Native 9-Core SNN")

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0
    
    # ==========================================
    # 🔥 Training Phase (Inner Loop)
    # ==========================================
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        sinabs.utils.reset_states(model.spiking_backbone)
        
        with autocast(device_type='cuda'):
            output = model(data)
            loss = criterion(output, target)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        batch_loss_val = loss.item()
        
        if np.isnan(batch_loss_val) or np.isinf(batch_loss_val):
            print(f"🚨 [CATASTROPHIC FAILURE] Batch {i+1} exploded into NaN! Aborting run.", flush=True)
            break
            
        total_loss += batch_loss_val

        current_time = time.time()
        if current_time - last_heartbeat_time >= 120:
            elapsed_total = current_time - global_start_time
            progress = (i + 1) / total_train_batches * 100
            print(f"   ⏳ [TRAIN HEARTBEAT] Epoch {epoch+1:02d} | Progress: {progress:.1f}% ({i+1}/{total_train_batches}) | Batch Loss: {batch_loss_val:.4f} | Job Runtime: {elapsed_total/60:.1f}m", flush=True)
            last_heartbeat_time = current_time

    # --- Compute Epoch Performance Statistics ---
    epoch_duration = time.time() - epoch_start_time
    avg_train_loss = total_loss / total_train_batches

    if np.isnan(avg_train_loss) or np.isinf(avg_train_loss):
        break

    # ==========================================
    # 🧪 Validation Phase (End of Epoch)
    # ==========================================
    model.eval()
    val_preds, val_targets = [], []
    total_val_batches = len(val_loader)
    
    with torch.no_grad():
        for v_idx, (data, target) in enumerate(val_loader):
            data = data.to(DEVICE)
            sinabs.utils.reset_states(model.spiking_backbone)
            output = model(data)
            preds = output.argmax(1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(target.numpy())
            
            current_time = time.time()
            if current_time - last_heartbeat_time >= 120:
                elapsed_total = current_time - global_start_time
                val_progress = (v_idx + 1) / total_val_batches * 100
                print(f"   ⏳ [VAL HEARTBEAT] Epoch {epoch+1:02d} | Val Progress: {val_progress:.1f}% ({v_idx+1}/{total_val_batches}) | Job Runtime: {elapsed_total/60:.1f}m", flush=True)
                last_heartbeat_time = current_time
    
    val_acc = 100. * np.sum(np.array(val_preds) == np.array(val_targets)) / len(val_targets)
    
    # 💾 Save weights ONLY if this beats your previous milestone checkpoint
    is_best = False
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        is_best = True
        save_plots(val_targets, val_preds, f"Epoch_{epoch+1}")

    # 📊 Hard Epoch Summary
    best_marker = " 🌟 [SAVED NEW BEST!]" if is_best else ""
    print(f"⏱️ Epoch {epoch+1:02d}/{EPOCHS} | Avg Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}% | Epoch Time: {epoch_duration:.1f}s{best_marker}", flush=True)

    scheduler.step()


# ==========================================
# 🏆 --- 5. FINAL UNBIASED TEST EVALUATION ---
# ==========================================
print("\n🔍 Training complete. Loading best checkpoint for final validation testing...", flush=True)

# 🎯 FIX: Explicitly match the unique source variable name here as well
if os.path.exists(best_model_path) and best_val_acc > 80.00:
    model.load_state_dict(torch.load(best_model_path))
else:
    print("ℹ️ No new epoch beat the 80.00% baseline. Falling back to original pretrained checkpoint.")
    model.load_state_dict(torch.load(PRETRAINED_SOURCE_PATH))

model.eval()

test_preds, test_targets = [], []
total_test_batches = len(test_loader)

print(f"🚀 Running Unbiased Test Set Evaluation ({total_test_batches} batches)...", flush=True)

with torch.no_grad():
    for t_idx, (data, target) in enumerate(test_loader):
        data = data.to(DEVICE)
        sinabs.utils.reset_states(model.spiking_backbone)
        output = model(data)
        preds = output.argmax(1).cpu().numpy()
        test_preds.extend(preds)
        test_targets.extend(target.numpy())
        
        current_time = time.time()
        if current_time - last_heartbeat_time >= 120:
            elapsed_total = current_time - global_start_time
            test_progress = (t_idx + 1) / total_test_batches * 100
            print(f"   ⏳ [TEST HEARTBEAT] Final Test Progress: {test_progress:.1f}% ({t_idx+1}/{total_test_batches}) | Job Runtime: {elapsed_total/60:.1f}m", flush=True)
            last_heartbeat_time = current_time

test_acc = 100. * np.sum(np.array(test_preds) == np.array(test_targets)) / len(test_targets)
print(f"\n🏆 FINAL UNBIASED TEST ACCURACY: {test_acc:.2f}%")

save_plots(test_targets, test_preds, "Final Test Validation")
print("\n📝 Final Test Classification Report:")
print(classification_report(test_targets, test_preds, target_names=['Flat', 'Curved'], zero_division=0), flush=True)