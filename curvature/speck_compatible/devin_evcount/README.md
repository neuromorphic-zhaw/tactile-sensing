# devin_evcount — count-binned models (THE CHAMPION FAMILY)

Bins hold ~equal event counts instead of equal time — "decide every N
events", the deployment-natural rasterization (the async chip has no
wall clock). Data: `data/count_bins/dataset_curv_evT166` etc.

Naming: `evT<B>_s<S>` = **T**otal bins/clip, forward stride ::**S**
(e.g. evT166_s2 → 83 steps). `evN<E>_s<S>` = **E**vents per bin
(opposite parameterization). `*_ft` = fine-tune of a named ckpt.

| Variant | Event-level result |
|---|---|
| `evT166_s2_ft` | **76.67 val / 70.00 test — champion** (200K windowed vote) |
| `evT166_s2` | 73.33 / 70.00 (vote) — also the warm-start backbone |
| `evT166_s1`, `evT166_c0.8/c0.9` | 56.67–66.67 val |
| `wclamp_evT166ft` | 66.67 val — binned v96.67 did NOT transfer |
| `evT332_s4`, `evN*` | unrunnable (deadlock) or untested |
