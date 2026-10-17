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
BATCH_SIZE = 2           # Reduced to 2 to fit the bigger model in VRAM
NUM_WORKERS = 4          # Slightly reduced to avoid CPU overhead with small batch
EPOCHS = 50
LEARNING_RATE = 2e-4 
ACCUMULATION_STEPS = 8    # 8 batches * 2 size = Effective Batch of 16
DEVICE = torch.device("cuda")
MAX_TIMESTEPS = 1000  

# --- 2. DATASET ---
class BinnedNPZDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, max_timesteps=1000, train=True): # <--- Added 'train=True'
        self.file_paths = file_paths
        self.labels = labels
        self.max_timesteps = max_timesteps
        self.train = train # <--- Now this works correctly

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        with np.load(self.file_paths[index]) as data:
            frames = data['frames'] 
            
        # 1. Temporal Jitter (KEEP THIS: still protects against approach/retreat timing)
        if self.train and len(frames) > self.max_timesteps:
            start = np.random.randint(0, len(frames) - self.max_timesteps)
            frames = frames[start : start + self.max_timesteps]
        else:
            frames = frames[:self.max_timesteps]

        # 2. Padding
        if len(frames) < self.max_timesteps:
            pad_amount = self.max_timesteps - len(frames)
            frames = np.pad(frames, ((0, pad_amount), (0, 0), (0, 0), (0, 0)), mode='constant')

        x = torch.from_numpy(frames).float()

        # 3. POLISH AUGMENTATION (Simplified for Precision)
        if self.train:
            # We remove flips and noise to allow the model to focus on clean geometry.
            # We reduce shift to +/- 1 pixel just to provide a tiny bit of variety.
            shift_h = np.random.randint(-1, 2)
            shift_w = np.random.randint(-1, 2)
            x = torch.roll(x, shifts=(shift_h, shift_w), dims=(2, 3))
            
        return x, self.labels[index]

# File identification
flat_files = [os.path.join(DATA_PATH, "flat", f) for f in os.listdir(os.path.join(DATA_PATH, "flat")) if f.endswith('.npz')]
curved_files = [os.path.join(DATA_PATH, "curved", f) for f in os.listdir(os.path.join(DATA_PATH, "curved")) if f.endswith('.npz')]
all_filepaths = flat_files + curved_files
labels = [0] * len(flat_files) + [1] * len(curved_files)

# Splits
indices = list(range(len(all_filepaths)))
train_val_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=labels, random_state=42)
train_val_labels = [labels[i] for i in train_val_idx]
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.176, stratify=train_val_labels, random_state=42)

train_dataset = BinnedNPZDataset(all_filepaths, labels, max_timesteps=MAX_TIMESTEPS, train=True)
val_dataset   = BinnedNPZDataset(all_filepaths, labels, max_timesteps=MAX_TIMESTEPS, train=False)

LOADER_ARGS = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 3}
train_loader = DataLoader(Subset(train_dataset, train_idx), shuffle=True, **LOADER_ARGS)
val_loader   = DataLoader(Subset(val_dataset, val_idx), shuffle=False, **LOADER_ARGS)

# --- 3. ARCHITECTURE ---
# --- ARCHITECTURE (Updated for flow) ---
class SpeckTactileSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False), 
            nn.BatchNorm2d(32), 
            sl.LIF(tau_mem=10.0), # Faster leak
            nn.MaxPool2d(2),      # Stronger signal

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
        # 1. Reset states for every new sample
        sinabs.utils.reset_states(self.model)
        
        batch_size, timesteps, C, H, W = x.shape
        
        # 2. Reshape for parallel processing
        x = x.reshape(batch_size * timesteps, C, H, W)
        
        # 3. Forward pass
        output = self.model(x)
        
        # 4. Reshape back and sum spikes over time
        output = output.reshape(batch_size, timesteps, -1)
        return output.sum(dim=1)

# Clear the "ghosts" from the GPU
gc.collect()
torch.cuda.empty_cache()

MAX_LR = 4e-4
model = SpeckTactileSNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=MAX_LR/10)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    steps_per_epoch=len(train_loader), # This tells it how many batches are in an epoch
    epochs=EPOCHS,
    pct_start=0.1,    
    anneal_strategy='cos'
)

# --- NEW: ROBUST LOAD LOGIC ---
MODEL_NAME = "best_4_layer_model"
best_val_acc = 0 # Initialize

# Check if any version of the 4-layer model exists
existing_files = [f for f in os.listdir('.') if f.startswith(MODEL_NAME) and f.endswith('.pt')]

if existing_files:
    # Sort to find the one with the highest accuracy in the filename
    latest_model = sorted(existing_files)[-1]
    print(f"📂 Found previous 4-layer models. Attempting to load: {latest_model}")
    checkpoint = torch.load(latest_model)
    
    try:
        model.load_state_dict(checkpoint, strict=True)
        # Extract accuracy from filename if possible (e.g., "best_4_layer_model_acc83.33.pt")
        try:
            best_val_acc = float(latest_model.split('acc')[-1].replace('.pt', ''))
        except:
            best_val_acc = 0
        print(f"✅ Successfully loaded weights. Resuming from {best_val_acc}% accuracy.")
    except:
        print("⚠️ Architecture mismatch (likely the old 2-layer model). Starting FRESH for 4-Layer brain.")
else:
    print("⚠️ No 4-layer model found. Starting from scratch.")

    
def plot_confusion_matrix(y_true, y_pred, epoch):
    # (Same plotting code as before)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Flat', 'Curved'], yticklabels=['Flat', 'Curved'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.savefig(f'confusion_matrix_epoch_{epoch+1}.png')
    plt.close()

# --- 4. TRAINING & VALIDATION LOOP ---
print(f"🚀 REFINEMENT START: 10ms Bins | Starting from {best_val_acc}%")

start_time = time.time()
# --- 4. TRAINING LOOP (With Accumulation) ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True) # Reset once at start of epoch
    
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # 1. Forward pass with Autocast
        with autocast(device_type='cuda'):
            output = model(data)
            # Scale loss by accumulation steps
            loss = criterion(output, target) / ACCUMULATION_STEPS 
            
        # 2. Backward pass (Accumulates gradients, doesn't clear them)
        scaler.scale(loss).backward()
        
        # 3. Only update weights every ACCUMULATION_STEPS
        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
        total_loss += loss.item() * ACCUMULATION_STEPS
        
        if i % 5 == 0:
            print(f"  Batch {i}/{len(train_loader)} | Loss: {loss.item()*ACCUMULATION_STEPS:.4f}")

    print(f"  ✅ Training Epoch {epoch+1} done. Running Validation...", flush=True)            
            
    # Validation Phase
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(DEVICE)
            output = model(data)
            preds = output.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.numpy())
    
    val_acc = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    # Update the learning rate based on validation performance
    # scheduler.step(val_acc) 

    # Log the current LR to see it dropping
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} | Val Acc: {val_acc:.2f}% | Current LR: {current_lr:.6f}")
    duration = time.time() - start_time
    
    # Logging & Saving
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{MODEL_NAME}_acc{val_acc:.2f}.pt")
        print(f"🌟 New Best Acc: {val_acc:.2f}%")
        plot_confusion_matrix(all_targets, all_preds, epoch)
        print(classification_report(all_targets, all_preds, target_names=['Flat', 'Curved'], zero_division=0))

    print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Val: {val_acc:.2f}% | {duration:.1f}s")

print(f"\n🏆 Final Best Accuracy: {best_val_acc:.2f}%")