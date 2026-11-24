import os
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch.amp import GradScaler
import tonic
import tonic.transforms as T
from sklearn.model_selection import train_test_split
import sinabs
import sinabs.layers as sl

# --- 1. CONFIGURATION (THE FAST LEVERS) ---
DATA_PATH = "/home/ronald/speck_project/curv/dataset_curv"
BATCH_SIZE = 64            
NUM_WORKERS = 8             
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda")

# FAST LEVER 1: Lock the time dimension to 10 bins!
transform = T.Compose([
    T.ToFrame(sensor_size=(128, 128, 2), n_time_bins=10),
])

# 2.1 Identify folder paths
flat_dir = os.path.join(DATA_PATH, "flat")
curved_dir = os.path.join(DATA_PATH, "curved")

flat_files = [os.path.join(flat_dir, f) for f in os.listdir(flat_dir) if f.endswith('.h5')]
curved_files = [os.path.join(curved_dir, f) for f in os.listdir(curved_dir) if f.endswith('.h5')]

all_filepaths = flat_files + curved_files
labels = [0] * len(flat_files) + [1] * len(curved_files)

class CustomTonicDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('t', 'i8'), ('p', 'i4')])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        with h5py.File(self.file_paths[index], 'r') as f:
            x = f['x'][:].astype('i4')
            y = f['y'][:].astype('i4')
            t = f['t'][:].astype('i8')
            p = f['p'][:].astype('i4')

        events = np.empty(len(x), dtype=self.dt)
        events['x'], events['y'], events['t'], events['p'] = x, y, t, p

        if self.transform:
            try:
                events = self.transform(events)
            except TypeError:
                events = self.transform(events.copy())
        return events, self.labels[index]                

full_dataset = CustomTonicDataset(all_filepaths, labels, transform=transform)

# FAST LEVER 2: Cache the dataset so Epoch 2+ is instant
# (Make sure you have a few GB of space on your drive!)
full_dataset = tonic.DiskCachedDataset(full_dataset, cache_path='./tactile_cache_fast')

# Splits
indices = list(range(len(full_dataset)))
train_val_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=labels, random_state=42)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.176, stratify=[labels[i] for i in train_val_idx], random_state=42)

# FAST LEVER 3: No more padding needed because every sample is exactly 10 bins long!
LOADER_ARGS = {
    'batch_size': BATCH_SIZE,
    'num_workers': NUM_WORKERS,
    'pin_memory': True, 
}

train_loader = DataLoader(Subset(full_dataset, train_idx), shuffle=True, **LOADER_ARGS)
val_loader   = DataLoader(Subset(full_dataset, val_idx), shuffle=False, **LOADER_ARGS)

# --- 3. ARCHITECTURE ---
class SpeckTactileSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False), 
            nn.BatchNorm2d(32),        # <-- FIX 1: Auto-scales voltage perfectly
            sl.LIF(tau_mem=20.0), 
            nn.MaxPool2d(2),           # <-- FIX 2: Preserves binary spikes!
            
            nn.Conv2d(32, 64, 3, padding=1, bias=False), 
            nn.BatchNorm2d(64),
            sl.LIF(tau_mem=20.0), 
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1, bias=False), 
            nn.BatchNorm2d(128),
            sl.LIF(tau_mem=20.0), 
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512, bias=False), 
            nn.BatchNorm1d(512),       # <-- 1D Batch Norm for Linear layers
            sl.LIF(tau_mem=20.0),
            
            nn.Linear(512, 2, bias=False)
        )

    def forward(self, x):
        sinabs.utils.reset_states(self.model)
        out = []
        for t in range(x.shape[1]):
            out.append(self.model(x[:, t]))
        return torch.stack(out).sum(dim=0)

model = SpeckTactileSNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

# --- 4. TRAINING LOOP ---
best_val_acc = 0
print("🚀 LAUNCHING 'GOLDILOCKS' SNN SCRIPT...")

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 🚫 REMOVED the * 50.0 boost. Let BatchNorm do the heavy lifting!
        data = data.to(DEVICE, non_blocking=True).float() 
        target = target.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True) 
        
        output = model(data)
        loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(DEVICE).float() # 🚫 Removed boost here too
            target = target.to(DEVICE)
            output = model(data)
            correct += output.argmax(1).eq(target).sum().item()
    
    val_acc = 100. * correct / len(val_idx)
    duration = time.time() - start_time
    
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Time: {duration:.2f}s")
print("\n" + "="*30)
print(f"🎉 Training Complete!")
print(f"🏆 Best Accuracy: {best_val_acc:.2f}%")
print(f"📅 Achieved at Epoch: {best_epoch}")
print("="*30)
