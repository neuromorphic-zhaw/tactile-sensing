import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import tonic
import tonic.transforms as T
from torch.utils.data import DataLoader, Subset
import sinabs
import sinabs.layers as sl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
DATA_PATH = "./dataset_curv" 
BINS = 80
DEVICE = torch.device("cuda")
MODEL_PATH = "speck_best_bins80_batch16_acc76.67.pt"

# --- 2. ARCHITECTURE ---
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
        out = []
        for t in range(x.shape[1]):
            out.append(self.model(x[:, t]))
        return torch.stack(out).sum(dim=0)

# --- 3. DATASET SETUP ---
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
            x, y, t, p = f['x'][:], f['y'][:], f['t'][:], f['p'][:]
        events = np.empty(len(x), dtype=self.dt)
        events['x'], events['y'], events['t'], events['p'] = x, y, t, p
        if self.transform:
            events = self.transform(events)
        return events, self.labels[index]

transform = T.Compose([T.ToFrame(sensor_size=(128, 128, 2), n_time_bins=BINS)])

# Identify files and split
flat_dir, curved_dir = os.path.join(DATA_PATH, "flat"), os.path.join(DATA_PATH, "curved")
flat_files = [os.path.join(flat_dir, f) for f in os.listdir(flat_dir) if f.endswith('.h5')]
curved_files = [os.path.join(curved_dir, f) for f in os.listdir(curved_dir) if f.endswith('.h5')]
all_filepaths = flat_files + curved_files
labels = [0] * len(flat_files) + [1] * len(curved_files)

full_dataset = CustomTonicDataset(all_filepaths, labels, transform=transform)
full_dataset = tonic.DiskCachedDataset(full_dataset, cache_path='./tactile_cache_80bins')

indices = list(range(len(full_dataset)))
_, test_idx = train_test_split(indices, test_size=0.15, stratify=labels, random_state=42)
test_set = Subset(full_dataset, test_idx)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# --- 4. RUN EVALUATION ---
print(f"Testing model: {MODEL_PATH}")
model = SpeckTactileSNN().to(DEVICE)

# Load the dictionary
state_dict = torch.load(MODEL_PATH)

# Remove any 'v_mem' entries so they don't cause size mismatches
keys_to_remove = [k for k in state_dict.keys() if 'v_mem' in k]
for k in keys_to_remove:
    del state_dict[k]

# Now load the cleaned dictionary
model.load_state_dict(state_dict, strict=False)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for data, targets in test_loader:
        outputs = model(data.to(DEVICE).float())
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.numpy())

# --- 5. SAVE MATRIX ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Flat (0)', 'Curved (1)'], 
            yticklabels=['Flat (0)', 'Curved (1)'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title(f'Confusion Matrix (80 Bins) - Acc: 76.67%')
plt.savefig('final_confusion_matrix.png')
print("✅ Success! Generated final_confusion_matrix.png")