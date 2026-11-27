# devin_10ms — 10ms time-binned models

Events rasterized into 10-ms frames (the original sweep family — pre
count-binning). Variant dirs carry their binned scores + recipe in the
name/README.txt: stride ::S, batch b, and aug/clamp deltas (wclamp,
stretch_aug, lr*, seed*, *_warm).

This was the binned-era champion family: wclamp v93.33/t86.67 — highest binned scores, but they did NOT transfer to event-level (~47–63% val). Deployment picks live in ../devin_evcount/.
