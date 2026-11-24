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
BATCH_SIZE = 2         
NUM_WORKERS = 8         
EPOCHS = 50
LEARNING_RATE = 1e-3    
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
LOADER_ARGS = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'pin_memory': True, 'persistent_workers': True}
train_loader = DataLoader(train_subset, shuffle=True, drop_last=True, **LOADER_ARGS)
val_loader   = DataLoader(val_subset, shuffle=False, drop_last=False, **LOADER_ARGS)
test_loader  = DataLoader(test_subset, shuffle=False, drop_last=False, **LOADER_ARGS)

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
        x = x.reshape(batch_size * timesteps, C, H, W)
        out_spikes = self.spiking_backbone(x)
        out_spikes = out_spikes.reshape(batch_size, timesteps, -1)
        return out_spikes.mean(dim=1)


model = Speck9LayerTrainer(target_batch_size=BATCH_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
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
best_val_acc = 0
best_model_path = "best_10ms_speck_compatible.pt"
total_train_batches = len(train_loader)
global_start_time = time.time()
last_heartbeat_time = time.time()  # Master clock for the 2-minute updates

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
        
        # Flush internal membrane history states between batches
        sinabs.utils.reset_states(model.spiking_backbone)
        
        with autocast(device_type='cuda'):
            output = model(data)
            loss = criterion(output, target)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        # ⏱️ Master Live Update Check (Every 2 Minutes)
        current_time = time.time()
        if current_time - last_heartbeat_time >= 120:
            elapsed_total = current_time - global_start_time
            progress = (i + 1) / total_train_batches * 100
            print(f"   ⏳ [TRAIN HEARTBEAT] Epoch {epoch+1:02d} | Progress: {progress:.1f}% ({i+1}/{total_train_batches}) | Batch Loss: {loss.item():.4f} | Job Runtime: {elapsed_total/60:.1f}m", flush=True)
            last_heartbeat_time = current_time

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
            
            # ⏱️ Master Live Update Check (Every 2 Minutes during Val)
            current_time = time.time()
            if current_time - last_heartbeat_time >= 120:
                elapsed_total = current_time - global_start_time
                val_progress = (v_idx + 1) / total_val_batches * 100
                print(f"   ⏳ [VAL HEARTBEAT] Epoch {epoch+1:02d} | Val Progress: {val_progress:.1f}% ({v_idx+1}/{total_val_batches}) | Job Runtime: {elapsed_total/60:.1f}m", flush=True)
                last_heartbeat_time = current_time
    
    val_acc = 100. * np.sum(np.array(val_preds) == np.array(val_targets)) / len(val_targets)
    epoch_duration = time.time() - epoch_start_time
    
    # 💾 Save weights ONLY if this is the absolute best model so far
    is_best = False
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        is_best = True
        save_plots(val_targets, val_preds, f"Epoch_{epoch+1}")

    # 📊 Hard Epoch Summary
    best_marker = " 🌟 [SAVED NEW BEST!]" if is_best else ""
    print(f"⏱️ Epoch {epoch+1:02d}/{EPOCHS} | Avg Train Loss: {total_loss/total_train_batches:.4f} | Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}% | Epoch Time: {epoch_duration:.1f}s{best_marker}", flush=True)


# ==========================================
# 🏆 --- 5. FINAL UNBIASED TEST EVALUATION ---
# ==========================================
print("\n🔍 Training complete. Loading best checkpoint for final validation testing...", flush=True)
model.load_state_dict(torch.load(best_model_path))
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
        
        # ⏱️ Master Live Update Check (Every 2 Minutes during Final Test)
        current_time = time.time()
        if current_time - last_heartbeat_time >= 120:
            elapsed_total = current_time - global_start_time
            test_progress = (t_idx + 1) / total_test_batches * 100
            print(f"   ⏳ [TEST HEARTBEAT] Final Test Progress: {test_progress:.1f}% ({t_idx+1}/{total_test_batches}) | Job Runtime: {elapsed_total/60:.1f}m", flush=True)
            last_heartbeat_time = current_time

test_acc = 100. * np.sum(np.array(test_preds) == np.array(test_targets)) / len(test_targets)
print(f"\n🏆 FINAL UNBIASED TEST ACCURACY: {test_acc:.2f}%")

# Generate final assets
save_plots(test_targets, test_preds, "Final Test Validation")
print("\n📝 Final Test Classification Report:")
print(classification_report(test_targets, test_preds, target_names=['Flat', 'Curved'], zero_division=0), flush=True)