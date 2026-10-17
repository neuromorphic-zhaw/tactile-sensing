import torch
import numpy as np
import os
import sinabs
import sinabs.layers as sl
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
DATA_PATH = "/cfs/earth/scratch/matheron/tactile-sensing/data/dataset_curv_binned_10ms" 
MODEL_PATH = "/cfs/earth/scratch/matheron/tactile-sensing/4_layer/clean100epoch/clean100_4layer_model_acc96.67.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TIMESTEPS = 1000

# --- 2. DATASET DEFINITION (Standard) ---
class BinnedNPZDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, max_timesteps=1000, train=False):
        self.file_paths = file_paths
        self.labels = labels
        self.max_timesteps = max_timesteps
        self.train = train

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, index):
        with np.load(self.file_paths[index]) as data:
            frames = data['frames'][:self.max_timesteps]
        
        if len(frames) < self.max_timesteps:
            pad_amount = self.max_timesteps - len(frames)
            frames = np.pad(frames, ((0, pad_amount), (0, 0), (0, 0), (0, 0)), mode='constant')
        
        return torch.from_numpy(frames).float(), self.labels[index]

# --- 3. RECREATE YOUR EXACT SPLITS ---
flat_files = [os.path.join(DATA_PATH, "flat", f) for f in os.listdir(os.path.join(DATA_PATH, "flat")) if f.endswith('.npz')]
curved_files = [os.path.join(DATA_PATH, "curved", f) for f in os.listdir(os.path.join(DATA_PATH, "curved")) if f.endswith('.npz')]
all_filepaths = flat_files + curved_files
labels = [0] * len(flat_files) + [1] * len(curved_files)

indices = list(range(len(all_filepaths)))
# This block is identical to yours to ensure the same test set is used
train_val_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=labels, random_state=42)

test_dataset = BinnedNPZDataset(all_filepaths, labels, max_timesteps=MAX_TIMESTEPS, train=False)
test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=2, shuffle=False)

# --- 4. ARCHITECTURE ---
class SpeckTactileSNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 32, 3, padding=1, bias=False), 
            torch.nn.BatchNorm2d(32), 
            sl.LIF(tau_mem=10.0), 
            torch.nn.MaxPool2d(2), 
            torch.nn.Conv2d(32, 64, 3, padding=1, bias=False), 
            torch.nn.BatchNorm2d(64), 
            sl.LIF(tau_mem=10.0), 
            torch.nn.MaxPool2d(2), 
            torch.nn.Conv2d(64, 128, 3, padding=1, bias=False), 
            torch.nn.BatchNorm2d(128), 
            sl.LIF(tau_mem=10.0), 
            torch.nn.MaxPool2d(2), 
            torch.nn.Conv2d(128, 128, 3, padding=1, bias=False), 
            torch.nn.BatchNorm2d(128), 
            sl.LIF(tau_mem=10.0), 
            torch.nn.MaxPool2d(2), 
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 8 * 8, 2, bias=False) 
        )

    def forward(self, x):
        sinabs.utils.reset_states(self.model)
        batch_size, timesteps, C, H, W = x.shape
        x = x.reshape(batch_size * timesteps, C, H, W)
        output = self.model(x)
        output = output.reshape(batch_size, timesteps, -1)
        return output.sum(dim=1)

# --- 5. LOADING & EVALUATION ---
model = SpeckTactileSNN().to(DEVICE)
print(f"📂 Loading Champion: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
# Filter state buffers to ensure clean loading
filtered_dict = {k: v for k, v in checkpoint.items() if "weight" in k or "bias" in k or "running" in k}
model.load_state_dict(filtered_dict, strict=False)
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for data, target in test_loader:
        output = model(data.to(DEVICE))
        all_preds.extend(output.argmax(1).cpu().numpy())
        all_targets.extend(target.numpy())

# --- 6. FINAL RESULTS ---
acc = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
print("\n" + "═"*40)
print(f"🏆 FINAL TEST SET ACCURACY: {acc:.2f}%")
print("═"*40)
print(classification_report(all_targets, all_preds, target_names=['Flat', 'Curved']))

# Confusion Matrix Plot
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Flat', 'Curved'], yticklabels=['Flat', 'Curved'])
plt.title(f'Test Set Results (Acc: {acc:.2f}%)')
plt.savefig('test_set_final_results.png')
print("✅ Results and Confusion Matrix saved to disk.")