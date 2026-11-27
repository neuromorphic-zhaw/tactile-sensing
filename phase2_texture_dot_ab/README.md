# phase2_texture_dot_ab — dot vs markerless gel A/B study

Sub-project: **multi-class surface/texture classification** (flat / curved /
edge / …) on the Speck2f + GelSight rig, run as a controlled A/B comparison:

- **Condition A (`dot`)** — GelSight gel **with** printed marker dots
- **Condition B (`nodot`)** — markerless gel (the current dataset gel)

Question: how much do marker dots contribute to event-based classification
accuracy? Dots add dense micro-motion events under shear/contact — the
hypothesis is `dot` outperforms `nodot`, with the gain concentrated in the
geometric classes (curved, edge).

## Controlled-experiment rules

Only the gel membrane may differ between the two conditions — and only in
the printed marker dots. Both membranes are the **same gel type**
(same geometry and elastomer hardness, used throughout the thesis);
condition B is simply an identical gel without dots printed on it.
Everything else is held constant — same sensor, same illumination, same
surface set, same press protocol, same clip duration, same rasterization
(evT166 count-bins), same network, same seeds/splits. Report results as a
paired delta.

Confounds to control at collection time:

- Both membranes are the same gel type — residual confound is limited to
  unit-to-unit manufacturing variation; note the gel serial/batch of each
  membrane in `RESULTS.md`.
- **The dots are hand-drawn with a marker** — they smudge, fade, and are
  redrawn between collection sessions, so the dot *pattern* is not fixed.
  The `dot` condition therefore tests **generalization to marker-type
  patterns**, not one specific layout. Redraw/refresh the dots between
  sessions (or clip batches) so the model learns "dots present" rather
  than memorizing a single pattern, and log each redraw in `RESULTS.md`.
- Match **illumination** between sessions; log any LED/lighting change.
- Use the **same set of stimulus surfaces**, ideally in the same numbered
  order under both gels — clip *i* in `dot/` then pairs with clip *i* in
  `nodot/`, enabling a paired test in `eval_compare.py`.

## Classes

Default classes: `flat`, `curved`, `edge`, `soft`, `grid`. Extend at
collection time with `--classes flat,curved,edge,soft,grid,ridge,...` — the
collector creates the folder layout and the trainer/eval read the label map
from `labels.json`. Keep the class list identical across both conditions.

## Data layout

```
data/phase2_texture_dot_ab/dataset_tex/{dot,nodot}/<class>/<date>_<class>_<nn>.h5   raw clips
data/phase2_texture_dot_ab/dataset_tex_evT166/{dot,nodot}/<class>/*.npz             evT166 frames+label
runs/<condition>_evT166/                                      trainer output
```

## Runbook

```bash
# 1. Collect — guided recorder: pick gel + class, then 4 spacebar presses
#    per clip: SPACE=start/approach → SPACE=touch (10.00 s precise timer)
#    → SPACE=retreat → SPACE=stop+save. Repeat each class in order, once
#    per condition (keep surface set + order identical across gels).
python record_touch.py --root ../data/phase2_texture_dot_ab/dataset_tex --classes flat,curved,edge,soft,grid
# batch alternative: python collect_tex.py --condition dot   --classes flat,curved,edge,soft,grid
#                    python collect_tex.py --condition nodot --classes flat,curved,edge,soft,grid

# 2. Rasterize to evT166 count-binned npz (per condition)
python prep_count_bins.py --src ../data/phase2_texture_dot_ab/dataset_tex/dot \
    --dst ../data/phase2_texture_dot_ab/dataset_tex_evT166/dot --bins 166
python prep_count_bins.py --src ../data/phase2_texture_dot_ab/dataset_tex/nodot \
    --dst ../data/phase2_texture_dot_ab/dataset_tex_evT166/nodot --bins 166

# 3. Train the identical model on each condition (same seed)
python train_tex.py --data ../data/phase2_texture_dot_ab/dataset_tex_evT166/dot   --out runs/dot_evT166
python train_tex.py --data ../data/phase2_texture_dot_ab/dataset_tex_evT166/nodot --out runs/nodot_evT166
# HPC: parameterized slurm launcher — one variant table, both conditions
# bash sweep.sh          # submits every VARIANTS[] line for dot AND nodot
# single run: sbatch --export=ALL,COND=dot,TAG=evT166_s2 job_tex.slurm
# overrides:  EPOCHS BS LR WD SEED TSTRIDE PRETRAINED DATA PARTITION

# 4. Compare — per-class accuracy, deltas, confusion matrices, McNemar if paired
python eval_compare.py --dot_run runs/dot_evT166 --nodot_run runs/nodot_evT166
```

## Metrics

Primary: overall test accuracy per condition + paired delta (`dot − nodot`).
Secondary: per-class accuracy delta, confusion matrices, McNemar's test on
paired clips when the collection order was matched. Binned accuracy is a
training diagnostic — confirm any claimed difference with event-level
specksim (`phase1_curvature/demo/eval_specksim_events.py`) and chip replay
(`phase1_curvature/demo/replay_chip.py`) per the repo's deployment-selection rule.
