"""Eval-only: load best_speck_fitting.pt, report test acc (same seed-42 split)."""
import glob, os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sinabs, sinabs.layers as sl

DATA_PATH = "/cfs/earth/scratch/matheron/tactile-sensing/data/dataset_curv_padded_10ms"
BATCH_SIZE = 10
DEVICE = torch.device("cuda")
CKPT = "best_speck_fitting.pt"

class D(Dataset):
    def __init__(s, d): s.f = glob.glob(os.path.join(d, "**/*.npz"), recursive=True)
    def __len__(s): return len(s.f)
    def __getitem__(s, i):
        with np.load(s.f[i]) as l:
            return torch.from_numpy(l['frames']).float(), torch.tensor(int(l['label']))

ds = D(DATA_PATH)
n = len(ds); tr, va = int(.7*n), int(.15*n)
_, _, test = torch.utils.data.random_split(ds, [tr, va, n-tr-va], generator=torch.Generator().manual_seed(42))
dl = DataLoader(test, batch_size=BATCH_SIZE)

def _iaf(b): return sl.IAFSqueeze(batch_size=b, spike_threshold=1.0, min_v_mem=-1.0,
                                 surrogate_grad_fn=sinabs.activation.PeriodicExponential())
class M(nn.Module):
    def __init__(s, b):
        super().__init__()
        s.bb = nn.Sequential(
            nn.Conv2d(2,16,4,2,1,bias=False), _iaf(b), nn.Conv2d(16,64,4,2,1,bias=False), _iaf(b),
            nn.Conv2d(64,32,4,2,1,bias=False), _iaf(b), nn.Conv2d(32,32,3,1,1,bias=False), _iaf(b),
            nn.Conv2d(32,32,4,2,1,bias=False), _iaf(b), nn.Conv2d(32,64,4,2,1,bias=False), _iaf(b),
            nn.Conv2d(64,64,3,1,1,bias=False), _iaf(b), nn.Conv2d(64,64,3,1,1,bias=False), _iaf(b),
            nn.AvgPool2d(2,2), nn.Flatten(), nn.Linear(256,2,bias=False), _iaf(b))
    def forward(s, x):
        B,T,C,H,W = x.shape; x = x[:, ::4].reshape(B*x[:, ::4].shape[1], C, H, W)
        return s.bb(x).reshape(B, -1, 2).mean(1)

m = M(BATCH_SIZE).to(DEVICE)
with torch.no_grad(): m(torch.zeros(BATCH_SIZE,40,2,128,128,device=DEVICE))
sd = torch.load(CKPT, map_location=DEVICE)
m.load_state_dict({k:v for k,v in sd.items() if not k.endswith('.v_mem')}, strict=False)
m.eval()
preds, tgts = [], []
with torch.no_grad():
    for d,t in dl:
        sinabs.utils.reset_states(m.bb)
        preds += m(d.to(DEVICE)).argmax(1).cpu().tolist(); tgts += t.tolist()
acc = 100.*np.sum(np.array(preds)==np.array(tgts))/len(tgts)
print(f"EVAL-ONLY FINAL TEST ACCURACY: {acc:.2f}%  ({len(tgts)} clips)")
