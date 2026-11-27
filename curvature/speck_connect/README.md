# speck_connect — data collection (stage 1)

`collect_gzip_curv.py` — the D2 dataset collection GUI. Connects to the
Speck2f devkit (USB / usbipd from WSL), streams DVS events, and saves
gzip-compressed `.h5` clips (x/y/t/p + label attrs) — the contract
format. Produced `data/dataset_curv/` (200 clips: 100 flat + 100 curved).

Stage-2 collection lives in `../../texture_dot_ab/` (new guided
recorder with phase tagging).
