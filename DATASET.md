# Dataset — Deliverable D2 (Speck-touch)

**Download:** [speck-touch_dataset_curv_200clips.tar.gz](https://zhaw.sharepoint.com/:u:/r/sites/OASIS/Freigegebene%20Dokumente/GelSight%20Speck/speck-touch_dataset_curv_200clips.tar.gz?d=w7d0ce924b40a4796aa5bb0d23f7b33e3&csf=1&web=1&e=WizIi5) (587 MB, ZHAW SharePoint)

## Contents

200 raw event recordings (100 `flat/` / 100 `curved/` surface contacts)
captured through the markerless GelSight elastomer on the Speck2f devkit's
integrated 128×128 DVS — the training dataset for all models in this repo.

- One `.h5` (HDF5, internally gzip-compressed) per clip — raw `x, y, t, p`
  event stream + `power_t`/`power_v` on-chip power traces; filenames
  `YYYY-MM-DD_<class>_<idx>.h5` encode the label
- `DATASET_README.txt` inside the archive; `data_report/REPORT.md` with
  per-class stats, session balance, and shortcut-risk audit
- Format per contract D2: **HDF5** (.aedat/.bin/.hdf5 accepted)
- Recorded March–April 2026 with `speck_connect/collect_gzip_curv.py`

## Derived datasets

Generated on demand from the raw clips (scripts in `data/`):

- `dataset_curv_events/` — raw events as `.npz` (re-binning master)
- `time_bins/dataset_curv_padded_{1,2,5,10}ms` — fixed-time-window rasters
- `count_bins/` — equal-event-count binned sets (evT166 etc., current
  strongest training recipe)
