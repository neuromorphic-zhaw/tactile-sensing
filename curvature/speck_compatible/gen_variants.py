"""Generate all resolution-sweep variant dirs from the canonical trainer.

Each variant = its own dir under devin_<res>/ with train.py + job.slurm + README.txt.
Patches applied to the wclamp_stretch template: DATA_PATH, BATCH_SIZE, stride,
AUG_* flags, WEIGHT_DECAY, EPOCHS, PRETRAINED_SOURCE_PATH, STRATIFIED split.
Rawevent variants use a RawEventSpeckDataset (events -> bins in __getitem__).
"""
import os, re

ROOT = "/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible"
DATA = "/cfs/earth/scratch/matheron/tactile-sensing/data"
WCLAMP_CKPT = f"{ROOT}/devin_10ms/wclamp_v93.33_t86.67/run_1069317/best_speck_fitting.pt"

with open(f"{ROOT}/devin_10ms/wclamp_stretch_v90.00_t83.33/train.py") as f:
    TEMPLATE = f.read()

STRAT_BLOCK = '''# stratified 70/15/15 per class (dataset is balanced 100 flat + 100 curved)
flat_idx  = [i for i, f in enumerate(full_dataset.file_list) if '/flat/' in f]
curv_idx  = [i for i, f in enumerate(full_dataset.file_list) if '/curved/' in f]
_g = torch.Generator().manual_seed(42)
_perm = lambda L: [L[i] for i in torch.randperm(len(L), generator=_g).tolist()]
_fp, _cp = _perm(flat_idx), _perm(curv_idx)
_n = lambda p: (int(0.70*len(p)), int(0.15*len(p)))
_ft, _fv = _n(_fp); _ct, _cv = _n(_cp)
train_subset = torch.utils.data.Subset(full_dataset, _fp[:_ft] + _cp[:_ct])
val_subset   = torch.utils.data.Subset(full_dataset, _fp[_ft:_ft+_fv] + _cp[_ct:_ct+_cv])
test_subset  = torch.utils.data.Subset(full_dataset, _fp[_ft+_fv:] + _cp[_ct+_cv:])
'''

RANDOM_BLOCK = '''train_subset, val_subset, test_subset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)'''

# ---- raw-event dataset: replaces PaddedSpeckTactileDataset ----
RAWEVENT_CLASS = '''class RawEventSpeckDataset(Dataset):
    """Streams raw {x,y,t,p} events; bins to BIN_US µs windows on the fly,
    pads to T_PAD bins. Event-level augs applied BEFORE binning (train only)."""
    BIN_US = 10000          # bin width in µs
    T_PAD = 1331            # padded bin count
    EV_JITTER_PX = 0        # spatial jitter amplitude (px)
    EV_DROP = 0.0           # random event drop probability
    EV_TJIT_US = 0          # timestamp jitter sigma (µs)
    EV_WARP = False         # piecewise-nonlinear time warp
    EV_POL = False          # 20% chance: drop one polarity entirely
    EV_MIXRES = 0           # >0: random bin width in [BIN_US*lo, BIN_US*hi]/100

    def __init__(self, data_dir):
        self.file_list = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
        if len(self.file_list) == 0:
            raise FileNotFoundError(f"No .npz files in {data_dir}")
        print(f"RawEvent dataset: {len(self.file_list)} event files.")

    def __len__(self):
        return len(self.file_list)

    def _bin(self, x, y, t, p, bin_us):
        tb = (t // bin_us).astype(np.int64)
        tb = np.clip(tb, 0, self.T_PAD - 1)
        frames = np.zeros((self.T_PAD, 2, 128, 128), dtype=np.float32)
        np.add.at(frames, (tb, p.astype(np.int64), y.astype(np.int64), x.astype(np.int64)), 1)
        return frames

    def __getitem__(self, index):
        with np.load(self.file_list[index]) as d:
            x = d['x'].astype(np.int32).copy()
            y = d['y'].astype(np.int32).copy()
            t = d['t'].astype(np.int64).copy()
            p = d['p'].astype(np.int64).copy()
            label = torch.tensor(int(d['label']), dtype=torch.long)
        t = t - t.min()
        bin_us = self.BIN_US

        if torch.is_grad_enabled():
            if self.EV_MIXRES:
                lo = self.BIN_US * 0.5; hi = self.BIN_US * 2.0
                bin_us = int(np.random.uniform(lo, hi))
            if self.EV_WARP:   # piecewise speed warp: 3 segments, rates ~U(0.7,1.4)
                rates = np.random.uniform(0.7, 1.4, 3)
                cuts = np.array([0, 1, 2, 3]) / 3.0 * t.max()
                seg = np.clip(np.digitize(t, cuts[1:3]), 0, 2)
                t = (cuts[seg] + (t - cuts[seg]) * rates[seg]).astype(np.int64)
            if self.EV_TJIT_US:
                t = np.clip(t + np.random.randn(len(t)) * self.EV_TJIT_US, 0, None).astype(np.int64)
            if self.EV_JITTER_PX:
                x += np.random.randint(-self.EV_JITTER_PX, self.EV_JITTER_PX + 1, len(x))
                y += np.random.randint(-self.EV_JITTER_PX, self.EV_JITTER_PX + 1, len(y))
            if self.EV_POL and np.random.rand() < 0.2:
                keep_pol = np.random.randint(0, 2)
                m = p == keep_pol; x, y, t, p = x[m], y[m], t[m], p[m]
            if self.EV_DROP > 0:
                m = np.random.rand(len(t)) > self.EV_DROP
                x, y, t, p = x[m], y[m], t[m], p[m]
            inb = (x >= 0) & (x < 128) & (y >= 0) & (y < 128)
            x, y, t, p = x[inb], y[inb], t[inb], p[inb]

        data = torch.from_numpy(self._bin(x, y, t, p, bin_us))

        if torch.is_grad_enabled():
            if np.random.rand() > 0.5:
                data = torch.roll(data, shifts=int(np.random.randint(-12, 12)), dims=0)
            if AUG_STRETCH:
                s = float(np.random.uniform(0.8, 1.2))
                tt = data.shape[0]
                new_t = max(2, int(round(tt / s)))
                idx = torch.linspace(0, tt - 1, new_t).round().long().clamp(0, tt - 1)
                d2 = data[idx]
                if new_t < tt:
                    data = torch.cat([d2, torch.zeros((tt - new_t,) + d2.shape[1:], dtype=d2.dtype)], dim=0)
                else:
                    off = int(np.random.randint(0, new_t - tt + 1))
                    data = d2[off:off + tt]

        data = torch.clamp(data, 0.0, 1.0)
        return data, label'''

PADDED_CLASS_RE = re.compile(r'class PaddedSpeckTactileDataset\(Dataset\):.*?return data, label', re.S)

def make_train(spec):
    t = TEMPLATE
    t = re.sub(r'DATA_PATH = ".*"', f'DATA_PATH = "{spec["data"]}"', t)
    t = re.sub(r'BATCH_SIZE = \d+', f'BATCH_SIZE = {spec["batch"]}', t)
    t = re.sub(r'WEIGHT_DECAY = [\de.-]+', f'WEIGHT_DECAY = {spec.get("wd", "1e-4")}', t)
    t = re.sub(r'x = x\[:, ::\d+\]', f'x = x[:, ::{spec["stride"]}]', t)
    t = t.replace('AUG_FLIP = False', f'AUG_FLIP = {spec.get("flip", False)}')
    t = t.replace('AUG_SHIFT = False', f'AUG_SHIFT = {spec.get("shift", False)}')
    t = t.replace('AUG_DROP = False', f'AUG_DROP = {spec.get("drop", False)}')
    t = t.replace('AUG_STRETCH = True', f'AUG_STRETCH = {spec.get("stretch", True)}')
    if spec.get("warm"):
        t = t.replace('PRETRAINED_SOURCE_PATH = ""', f'PRETRAINED_SOURCE_PATH = "{WCLAMP_CKPT}"')
    if spec.get("strat"):
        t = t.replace(RANDOM_BLOCK, STRAT_BLOCK)
    if spec.get("rawevent"):
        t = PADDED_CLASS_RE.sub(RAWEVENT_CLASS, t, count=1)
        for k, v in spec["ev"].items():
            t = re.sub(rf'(\n    {k} = )[^\n]+', rf'\g<1>{v}', t)
        # rawevent: dataset class name swap
        t = t.replace('PaddedSpeckTactileDataset', 'RawEventSpeckDataset')
    return t

SLURM = '''#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={part}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={time}
#SBATCH --output={vd}/run_%j.out
#SBATCH --error={vd}/run_%j.err

export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
python -c "import sinabs; print('sinabs loaded')" || exit 1
rm -rf ~/.cache/triton/* /tmp/torchinductor_${{USER}}* ~/.cache/torch/kernels/*
ipcrm -a 2>/dev/null || true
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${{SLURM_JOB_ID}}"
mkdir -p $TORCH_EXTENSIONS_DIR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VD={vd}
RUN_DIR=$VD/run_${{SLURM_JOB_ID}}
mkdir -p "$RUN_DIR"
cp "$VD/job.slurm" "$RUN_DIR/"
cp "$VD/train.py" "$RUN_DIR/"
cd "$RUN_DIR"
echo "Run dir: $RUN_DIR"
python -u "$VD/train.py"
# move slurm logs inside the run folder so the variant dir stays clean
mv "$VD/run_${{SLURM_JOB_ID}}.out" "$VD/run_${{SLURM_JOB_ID}}.err" "$RUN_DIR/" 2>/dev/null || true
rm -rf /tmp/torchinductor_${{USER}}* ~/.cache/triton/* $TORCH_EXTENSIONS_DIR
'''

# stack = stretch+drop+wclamp (wclamp is in-template already)
STACK = dict(stretch=True, drop=True)

VARIANTS = {
 "devin_5ms": dict(data=f"{DATA}/time_bins/dataset_curv_padded_5ms", runs=[
  ("s4_best",        dict(stride=4,  batch=10, **STACK), "earth-5", "8:00:00",  "stack ::4 666st ~17GB"),
  ("s4_b8",          dict(stride=4,  batch=8,  **STACK), "earth-5", "8:00:00",  "stack ::4 b8 ~14GB"),
  ("s2_best",        dict(stride=2,  batch=10, **STACK), "earth-4", "8:00:00",  "stack ::2 1331st ~33GB"),
  ("s2_strat",       dict(stride=2,  batch=10, strat=True, **STACK), "earth-4", "8:00:00", "::2 + stratified split"),
  ("s2_warm",        dict(stride=2,  batch=10, warm=True, stretch=True), "earth-4", "8:00:00", "::2 warmstart wclamp ckpt"),
  ("s1_b5_stream",   dict(stride=1,  batch=5,  **STACK), "earth-4", "20:00:00", "MAX STREAM ::1 b5 2663st ~33GB"),
 ]),
 "devin_2ms": dict(data=f"{DATA}/time_bins/dataset_curv_padded_2ms", runs=[
  ("s8_best",        dict(stride=8,  batch=10, **STACK), "earth-5", "8:00:00",  "stack ::8 833st ~21GB"),
  ("s8_strat",       dict(stride=8,  batch=10, strat=True, **STACK), "earth-5", "8:00:00", "::8 + stratified"),
  ("s8_wd",          dict(stride=8,  batch=10, wd="1e-3", **STACK), "earth-5", "8:00:00", "::8 + wd1e3"),
  ("s4_b5",          dict(stride=4,  batch=5,  **STACK), "earth-5", "8:00:00",  "stack ::4 b5 1665st ~21GB"),
  ("s4_best",        dict(stride=4,  batch=10, **STACK), "earth-4", "20:00:00", "stack ::4 b10 1665st ~41GB L40S-only"),
  ("s2_b4_stream",   dict(stride=2,  batch=4,  **STACK), "earth-4", "40:00:00", "MAX STREAM ::2 b4 3330st ~33GB"),
 ]),
 "devin_1ms": dict(data=f"{DATA}/time_bins/dataset_curv_padded_1ms", runs=[
  ("s16_best",       dict(stride=16, batch=10, **STACK), "earth-5", "8:00:00",  "stack ::16 833st ~21GB"),
  ("s8_b5",          dict(stride=8,  batch=5,  **STACK), "earth-5", "8:00:00",  "stack ::8 b5 1665st ~21GB"),
  ("s8_strat",       dict(stride=8,  batch=5,  strat=True, **STACK), "earth-5", "8:00:00", "::8 b5 + stratified"),
  ("s10_b10",        dict(stride=10, batch=10, **STACK), "earth-4", "8:00:00",  "::10 1332st ~33GB — 1ms granule on 10ms grid"),
  ("s5_b5",          dict(stride=5,  batch=5,  **STACK), "earth-4", "20:00:00", "stack ::5 b5 2664st ~33GB"),
  ("s4_b4_stream",   dict(stride=4,  batch=4,  **STACK), "earth-4", "40:00:00", "MAX STREAM ::4 b4 3330st ~33GB"),
 ]),
 "devin_rawevent": dict(data=f"{DATA}/dataset_curv_events", runs=[
  ("ev_base",  dict(stride=4, batch=10, rawevent=True, stretch=False, ev={"BIN_US":10000,"T_PAD":1331}), "earth-5", "8:00:00", "events->10ms bins ::4, parity anchor vs padded_10ms"),
  ("ev_augs",  dict(stride=4, batch=10, rawevent=True, stretch=True, ev={"BIN_US":10000,"T_PAD":1331,"EV_JITTER_PX":6,"EV_DROP":0.10,"EV_TJIT_US":2000}), "earth-5", "8:00:00", "event-level augs: jitter±6px, evdrop10%, tjit2ms"),
  ("ev_bin5ms",dict(stride=4, batch=10, rawevent=True, **STACK, ev={"BIN_US":5000,"T_PAD":2663}), "earth-5", "8:00:00", "events->5ms bins ::4 — resolution via events"),
  ("ev_mixres",dict(stride=4, batch=10, rawevent=True, **STACK, ev={"BIN_US":10000,"T_PAD":2700,"EV_MIXRES":1}), "earth-5", "8:00:00", "random binwidth 5-20ms/sample — res-invariance"),
  ("ev_pol",   dict(stride=4, batch=10, rawevent=True, **STACK, ev={"BIN_US":10000,"T_PAD":1331,"EV_POL":1}), "earth-5", "8:00:00", "polarity-drop aug 20%"),
  ("ev_warp",  dict(stride=4, batch=10, rawevent=True, **STACK, ev={"BIN_US":10000,"T_PAD":1331,"EV_WARP":1}), "earth-5", "8:00:00", "nonlinear piecewise timewarp"),
 ]),
 # --- event-COUNT-binned datasets (count_bins/) — deployment-faithful regime ---
 "devin_evcount": dict(data=None, runs=[
  ("evT332_s4",   dict(stride=4, batch=10, **STACK), "earth-5", "8:00:00", "evT332 ::4 = 83 steps — parity vs wclamp"),
  ("evT166_s2",   dict(stride=2, batch=10, **STACK), "earth-5", "8:00:00", "evT166 ::2 = 83 steps — coarser norm"),
  ("evN4000_s8",  dict(stride=8, batch=10, **STACK), "earth-5", "8:00:00", "evN4000 ::8 ~98 steps — deploy 'every 4K ev'"),
  ("evN2000_s8",  dict(stride=8, batch=10, **STACK), "earth-4", "8:00:00", "evN2000 ::8 ~195 steps"),
  ("evN2000_s4",  dict(stride=4, batch=10, **STACK), "earth-4", "8:00:00", "evN2000 ::4 ~390 steps — deeper"),
  ("evN1000_s16", dict(stride=16, batch=10, **STACK), "earth-4", "20:00:00", "evN1000 ::16 ~195 steps — finest"),
 ]),
}
# evcount variants: dataset dir varies per-run -> patch data path individually
EVCOUNT_DATA = f"{DATA}/count_bins"
EVCOUNT_DS = {"evT332_s4": "dataset_curv_evT332", "evT166_s2": "dataset_curv_evT166",
              "evN4000_s8": "dataset_curv_evN4000", "evN2000_s8": "dataset_curv_evN2000",
              "evN2000_s4": "dataset_curv_evN2000", "evN1000_s16": "dataset_curv_evN1000"}

for dset_dir, cfg in VARIANTS.items():
    for name, spec, part, tlimit, desc in cfg["runs"]:
        vd = f"{ROOT}/{dset_dir}/{name}"
        os.makedirs(vd, exist_ok=True)
        spec = dict(spec)
        spec["data"] = (f"{EVCOUNT_DATA}/{EVCOUNT_DS[name]}"
                        if cfg["data"] is None else cfg["data"])
        with open(f"{vd}/train.py", "w") as f:
            f.write(make_train(spec))
        with open(f"{vd}/job.slurm", "w") as f:
            f.write(SLURM.format(name=f"{dset_dir[6:]}_{name}", vd=vd, part=part, time=tlimit))
        with open(f"{vd}/README.txt", "w") as f:
            f.write(f"{desc}\n")
        print(f"built {dset_dir}/{name}")
print("ALL VARIANTS GENERATED")
