# texture_dot_ab — dot vs markerless gel A/B study

Sub-project: **multi-class surface/texture classification** (flat / curved /
edge / …) on the Speck2f + GelSight rig, run as a controlled A/B comparison:

- **Condition A (`dot`)** — GelSight gel **with** printed marker dots
- **Condition B (`nodot`)** — markerless gel (the current dataset gel)

Question: how much do marker dots contribute to event-based classification
accuracy? Dots add dense micro-motion events under shear/contact — the
hypothesis is `dot` outperforms `nodot`, with the gain concentrated in the
geometric classes (curved, edge).

## Controlled-experiment rules

Only the gel membrane may differ between the two conditions. Everything else
is held constant — same sensor, same illumination, same surface set, same
press protocol, same clip duration, same rasterization (evT166 count-bins),
same network, same seeds/splits. Report results as a paired delta.

Confounds to control at collection time:

- Both gels should have the **same geometry and elastomer hardness**; a
  different-thickness gel conflates membrane mechanics with the dot signal.
- Match **illumination** between sessions; log any LED/lighting change.
- Use the **same set of stimulus surfaces**, ideally in the same numbered
  order under both gels — clip *i* in `dot/` then pairs with clip *i* in
  `nodot/`, enabling a paired test in `eval_compare.py`.

## Classes

Default classes: `flat`, `curved`, `edge`. Extend at collection time with
`--classes flat,curved,edge,ridge,...` — the collector creates the folder
layout and the trainer/eval read the label map from `labels.json`. Keep the
class list identical across both conditions.

## Data layout

```
data/dataset_tex/{dot,nodot}/<class>/<date>_<class>_<nn>.h5   raw clips
data/dataset_tex_evT166/{dot,nodot}/<class>/*.npz             evT166 frames+label
runs/<condition>_evT166/                                      trainer output
```

## Runbook

```bash
# 1. Collect (run once per condition; keep surface set + order identical)
python collect_tex.py --condition dot   --classes flat,curved,edge
python collect_tex.py --condition nodot --classes flat,curved,edge

# 2. Rasterize to evT166 count-binned npz (per condition)
python prep_count_bins.py --src ../data/dataset_tex/dot \
    --dst ../data/dataset_tex_evT166/dot --bins 166
python prep_count_bins.py --src ../data/dataset_tex/nodot \
    --dst ../data/dataset_tex_evT166/nodot --bins 166

# 3. Train the identical model on each condition (same seed)
python train_tex.py --data ../data/dataset_tex_evT166/dot   --out runs/dot_evT166
python train_tex.py --data ../data/dataset_tex_evT166/nodot --out runs/nodot_evT166

# 4. Compare — per-class accuracy, deltas, confusion matrices, McNemar if paired
python eval_compare.py --dot_run runs/dot_evT166 --nodot_run runs/nodot_evT166
```

## Metrics

Primary: overall test accuracy per condition + paired delta (`dot − nodot`).
Secondary: per-class accuracy delta, confusion matrices, McNemar's test on
paired clips when the collection order was matched. Binned accuracy is a
training diagnostic — confirm any claimed difference with event-level
specksim (`demo/eval_specksim_events.py`) and chip replay
(`demo/replay_chip.py`) per the repo's deployment-selection rule.
