# devin_1ms — 1ms time-binned models

Events rasterized into 1-ms frames (the original sweep family — pre
count-binning). Variant dirs carry their binned scores + recipe in the
name/README.txt: stride ::S, batch b, and aug/clamp deltas (wclamp,
stretch_aug, lr*, seed*, *_warm).

Fine-grained binning ≠ better here; results are in the model registry. Deployment picks live in ../devin_evcount/.
