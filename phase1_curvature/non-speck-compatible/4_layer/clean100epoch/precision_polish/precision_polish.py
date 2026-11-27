import matplotlib
matplotlib.use('Agg') # Force headless mode
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import sinabs
import sinabs.layers as sl
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# --- 1. CONFIGURATION ---
DATA_PATH = "/cfs/earth/scratch/matheron/tactile-sensing/data/dataset_curv_binned_10ms" 
BATCH_SIZE = 2 
NUM_WORKERS = 4 
EPOCHS = 100             # Increased to 100 for deep refinement
ACCUMULATION_STEPS = 8 
DEVICE = torch.device("cuda")
MAX_TIMESTEPS = 1000  
MAX_LR = 5e-5            # Kept low to avoid destroying the champion weights

# --- 2. DATASET ---
class BinnedNPZDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, max_timesteps=1000, train=True):
        self.file_paths = file_paths
        self.labels = labels
        self.max_timesteps = max_timesteps
        self.train = train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        with np.load(self.file_paths[index]) as data:
            frames = data['frames'] 
            
        if self.train and len(frames) > self.max_timesteps:
            start = np.random.randint(0, len(frames) - self.max_timesteps)
            frames = frames[start : start + self.max_timesteps]
        else:
            frames = frames[:self.max_timesteps]

        if len(frames) < self.max_timesteps:
            pad_amount = self.max_timesteps - len(frames)
            frames = np.pad(frames, ((0, pad_amount), (0, 0), (0, 0), (0, 0)), mode='constant')

        x = torch.from_numpy(frames).float()

        if self.train:
            # Spatial Jitter (+/- 3 pixels)
            shift_h = np.random.randint(-3, 4)
            shift_w = np.random.randint(-3, 4)
            x = torch.roll(x, shifts=(shift_h, shift_w), dims=(2, 3))
            
        return x, self.labels[index]

# Data Splitting (Exact same split to maintain integrity of the 80% baseline)
flat_files = [os.path.join(DATA_PATH, "flat", f) for f in os.listdir(os.path.join(DATA_PATH, "flat")) if f.endswith('.npz')]
curved_files = [os.path.join(DATA_PATH, "curved", f) for f in os.listdir(os.path.join(DATA_PATH, "curved")) if f.endswith('.npz')]
all_filepaths = flat_files + curved_files
labels = [0] * len(flat_files) + [1] * len(curved_files)

indices = list(range(len(all_filepaths)))
train_val_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=labels, random_state=42)
train_val_labels = [labels[i] for i in train_val_idx]
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.176, stratify=train_val_labels, random_state=42)

train_dataset = BinnedNPZDataset(all_filepaths, labels, max_timesteps=MAX_TIMESTEPS, train=True)
val_dataset   = BinnedNPZDataset(all_filepaths, labels, max_timesteps=MAX_TIMESTEPS, train=False)
test_dataset  = BinnedNPZDataset(all_filepaths, labels, max_timesteps=MAX_TIMESTEPS, train=False)

LOADER_ARGS = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 3}
train_loader = DataLoader(Subset(train_dataset, train_idx), shuffle=True, **LOADER_ARGS)
val_loader   = DataLoader(Subset(val_dataset, val_idx), shuffle=False, **LOADER_ARGS)
test_loader  = DataLoader(Subset(test_dataset, test_idx), shuffle=False, **LOADER_ARGS)

# --- 3. ARCHITECTURE ---
class SpeckTactileSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False), 
            nn.BatchNorm2d(32), 
            sl.LIF(tau_mem=10.0), 
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, 3, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            sl.LIF(tau_mem=10.0), 
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, 3, padding=1, bias=False), 
            nn.BatchNorm2d(128), 
            sl.LIF(tau_mem=10.0), 
            nn.MaxPool2d(2), 
            nn.Conv2d(128, 128, 3, padding=1, bias=False), 
            nn.BatchNorm2d(128), 
            sl.LIF(tau_mem=10.0), 
            nn.MaxPool2d(2), 
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 2, bias=False) 
        )

    def forward(self, x):
        sinabs.utils.reset_states(self.model)
        batch_size, timesteps, C, H, W = x.shape
        x = x.reshape(batch_size * timesteps, C, H, W)
        output = self.model(x)
        output = output.reshape(batch_size, timesteps, -1)
        return output.sum(dim=1)

# --- OPTIMIZER & LOADING ---
gc.collect()
torch.cuda.empty_cache()

model = SpeckTactileSNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=MAX_LR)
# LABEL SMOOTHING (0.1) prevents the model from over-committing to training noise
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()

updates_per_epoch = len(train_loader) // ACCUMULATION_STEPS
total_updates = (updates_per_epoch * EPOCHS) + 20 # Increased buffer

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=MAX_LR, 
    total_steps=total_updates, 
    pct_start=0.3, # 30% warmup is better for 100-epoch runs
    anneal_strategy='cos'
)

MODEL_NAME = "final_100ep_refine"
CHAMPION_PATH = "/cfs/earth/scratch/matheron/tactile-sensing/4_layer/clean100epoch/clean100_4layer_model_acc96.67.pt"
best_val_acc = 80.0 # Ignore anything that doesn't at least beat our test baseline

if os.path.exists(CHAMPION_PATH):
    print(f"🚀 Initializing from Champion: {CHAMPION_PATH}")
    checkpoint = torch.load(CHAMPION_PATH)
    filtered_dict = {k: v for k, v in checkpoint.items() if "weight" in k or "bias" in k or "running" in k}
    model.load_state_dict(filtered_dict, strict=False)
else:
    print("⚠️ No Champion found at path. Starting Fresh.")

# --- 4. TRAINING LOOP ---
print(f"🚀 STARTING 100-EPOCH REFINEMENT | Save Threshold: >{best_val_acc}%")
best_model_path = ""

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)
    
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # 20% Spike Drop is the SNN Sweet Spot
        if model.training:
            data = data * (torch.rand(data.shape, device=DEVICE) > 0.20)
            
        with autocast(device_type='cuda'):
            output = model(data)
            loss = criterion(output, target) / ACCUMULATION_STEPS 
            
        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler.last_epoch < total_updates - 1:
                scheduler.step()
        
        total_loss += loss.item() * ACCUMULATION_STEPS
        
    # Validation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data.to(DEVICE))
            all_preds.extend(output.argmax(1).cpu().numpy())
            all_targets.extend(target.numpy())
    
    val_acc = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = f"{MODEL_NAME}_acc{val_acc:.2f}.pt"
        torch.save(model.state_dict(), best_model_path)
        print(f"🌟 High Score Found: {val_acc:.2f}% (Saved to {best_model_path})")

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")

# --- 5. AUTOMATIC TEST SET EVALUATION ---
if best_model_path and os.path.exists(best_model_path):
    print("\n" + "="*60)
    print("🏆 REFINEMENT COMPLETE. RUNNING FINAL UNSEEN TEST EVALUATION...")
    print(f"📦 Best Model: {best_model_path}")
    
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_preds, test_targets = [], []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(DEVICE))
            test_preds.extend(output.argmax(1).cpu().numpy())
            test_targets.extend(target.numpy())
            
    final_test_acc = 100. * np.sum(np.array(test_preds) == np.array(test_targets)) / len(test_targets)
    print("\n" + "💎"*20)
    print(f"👑 FINAL TEST ACCURACY: {final_test_acc:.2f}%")
    print("💎"*20)
    print(classification_report(test_targets, test_preds, target_names=['Flat', 'Curved']))
else:
    print("\n🛑 Training ended without beating the 80% baseline. No final test performed.")