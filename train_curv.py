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
DATA_PATH = "./dataset_curv" 
BATCH_SIZE = 4             # Keep low for 80 bins
NUM_WORKERS = 16            # Increased for faster gzip unpacking
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda")
BINS = 80                  # Your specific requirement

# FAST LEVER 1: Set to 80 bins
transform = T.Compose([
    T.ToFrame(sensor_size=(128, 128, 2), n_time_bins=BINS),
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

# Cache on scratch
full_dataset = tonic.DiskCachedDataset(full_dataset, cache_path='./tactile_cache_80bins')

# Splits
indices = list(range(len(full_dataset)))
train_val_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=labels, random_state=42)
train_val_labels = [labels[i] for i in train_val_idx]
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.176, stratify=train_val_labels, random_state=42)

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
            nn.BatchNorm2d(32),        
            sl.LIF(tau_mem=20.0), 
            nn.MaxPool2d(2),           
            
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
            nn.BatchNorm1d(512),       
            sl.LIF(tau_mem=20.0),
            
            nn.Linear(512, 2, bias=False)
        )

    def forward(self, x):
        sinabs.utils.reset_states(self.model)
        acc_output = None
        for t in range(x.shape[1]):
            out = self.model(x[:, t])
            if acc_output is None:
                acc_output = out
            else:
                acc_output = acc_output + out
        return acc_output

model = SpeckTactileSNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

# --- 4. TRAINING LOOP ---
best_val_acc = 0
best_model_path = ""

print(f"🚀 LAUNCHING 'GOLDILOCKS' SNN (Bins: {BINS}, Batch: {BATCH_SIZE})...")

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
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
            data = data.to(DEVICE).float()
            target = target.to(DEVICE)
            output = model(data)
            correct += output.argmax(1).eq(target).sum().item()
    
    val_acc = 100. * correct / len(val_idx)
    duration = time.time() - start_time
    
    # SAVE LOGIC
    if val_acc > best_val_acc:
        # Delete old best model file if it exists to save space
        if best_model_path and os.path.exists(best_model_path):
            os.remove(best_model_path)
            
        best_val_acc = val_acc
        best_model_path = f"speck_best_bins{BINS}_batch{BATCH_SIZE}_acc{val_acc:.2f}.pt"
        torch.save(model.state_dict(), best_model_path)
        print(f"🌟 New Best Model Saved: {best_model_path}")
    
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Time: {duration:.2f}s")
    torch.cuda.empty_cache()

print("\n" + "="*30)
print(f"🎉 Training Complete!")
print(f"🏆 Final Best Accuracy: {best_val_acc:.2f}%")
print(f"💾 File: {best_model_path}")
print("="*30)
