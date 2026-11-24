# Speck Mock Pipeline — convert, map & run BEFORE real training

Proves the entire deployment path end-to-end without a real dataset:
mock-train on synthetic frames → real `.pt` checkpoint → quantize →
solve layer→core mapping → generate full Samna config → run live on the
attached Speck2f devkit.

## Why this exists

`DynapcnnNetwork(...)` succeeding means NOTHING — layer-type checks only.
The real gate is the memory solver inside `make_config()`/`.to()`. The old
86.67% checkpoint passed every check and still failed with
`No valid mapping found`. This demo runs the real solver so nothing burns
GPU hours on an unmappable architecture again.

## Files

| file | what it does |
|---|---|
| `model.py` | parametric backbone, same forward contract as the real trainer |
| `maximize.py` | sweeps channel counts → finds the largest taper that maps |
| `mock_train.py` | synthetic data → real checkpoint `mock_speck.pt` |
| `verify_convert.py` | offline proof: convert + memory + map + make_config |
| `run_live.py` | deploy to devkit + live flat/curved/idle readout (+ `--flash`) |

## The maximum-fitting architecture (found by maximize.py)

```
MAX_CHANNELS = (16, 64, 32, 32, 32, 64, 64, 64)   # 182,290 params
```

| layer | shape | kernel | neuron | core |
|---|---|---|---|---|
| 1 | Conv 2→16 k4 s2 (128→64²) | 0.5 Ki | **64.0 Ki** | 0 |
| 2 | Conv 16→64 k4 s2 (64→32²) | 16.0 Ki | **64.0 Ki** | 1 |
| 3 | Conv 64→32 k4 s2 (32→16²) | 32.0 Ki | 8.0 Ki | 3 |
| 4 | Conv 32→32 k3 s1 (16²) | 16.0 Ki | 8.0 Ki | 2 |
| 5 | Conv 32→32 k4 s2 (16→8²) | 16.0 Ki | 2.0 Ki | 8 |
| 6 | Conv 32→64 k4 s2 (8→4²) | 32.0 Ki | 1.0 Ki | 4 |
| 7 | Conv 64→64 k3 s1 (4²) | **64.0 Ki** | 1.0 Ki | 6 |
| 8 | Conv 64→64 k3 s1 (4²) | **64.0 Ki** | 1.0 Ki | 5 |
| 9 | AvgPool→Linear 256→2 | 0.5 Ki | ~0 | 7 |

**Kernel memory is the wall**: cores 1,2,3,4,5,6,8 all sit at 100% kernel.
No conv layer can grow without another shrinking first. Rule for the real
trainer: never exceed these per-layer channel counts.

## Workflow

```bash
cd demo
python maximize.py                 # re-run the capacity sweep (optional)
python mock_train.py               # ~1-2 min on CPU -> mock_speck.pt
python verify_convert.py --model mock_speck.pt   # offline PASS/FAIL
python run_live.py --model mock_speck.pt --flash # real devkit + readout
```

Expected: `verify_convert` prints `PASS`; `run_live` prints
`deployed — layer->core map: {...}` then streams FLAT/CURVED/idle windows.
With a mock (untrained) checkpoint both features fire ~equally — a trained
checkpoint separates them.

## Notes

- `mock_train.py` val accuracy can sit at 50% on tiny synthetic data — the
  checkpoint format is what matters (27 tensors incl. v_mem buffers, same
  as the real trainer writes; `load_checkpoint` strips v_mem like deploy).
- `run_live.py` mirrors `deploy_speck.py` — same device string, slow clock,
  majority-vote readout, `io_sel=24` in the flash binary.
- `demo_speck_standalone.bin` is a real flashable image of the mock net.
