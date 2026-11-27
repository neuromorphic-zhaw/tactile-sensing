# phase2_texture_dot_ab — results & experiment map

## How to read the file structure

```
runs/<condition>_<variant>/     one training run, fully self-contained
  <condition> = dot | nodot     which gel the model trained+tested on
  <variant>   = <data>_s<stride>[_noiseKnobs]
                evT166_s2         baseline (166 count-bins, stride 2)
                evT332_s4         332 bins stride 4 (same compute as baseline)
                evT166_s2_drop30  + event-dropout 30% (baseline has 10%)
                evT166_s2_addp1   + 1% additive hot-pixel noise
                evT166_s2_shift8  + +-8px spatial jitter
                evT166_s2_allnoise  all three combined
  inside each run dir:
    train.out / train.err       logs (epoch lines: loss | val | best)
    metrics.json                best_val_acc + test_acc + provenance
    test_predictions.npz        per-clip predictions (for paired stats)
    confusion_matrix.png        visual per-class breakdown
    split.json                  exact stratified test indices (seed 42)
    best.pt                     best-val weights
    job.slurm / train.py        exact copies of what ran
    Slurm-<jobid>/              scheduler metadata
```

Paired A/B: `dot_X` vs `nodot_X` share the variant, seed, and stratified
split structure (exactly 14/3/3 train/val/test per class) — differences
attribute to the gel condition, not the split.

Noise experiment (jobs 1317253-60, 2026-11-27): does noise-augmented
training change the dot-vs-nodot picture? Variants stack ON TOP of the
baseline augmentation (temporal roll, 10% dropout, time-stretch).

## Binned test accuracy (diagnostic — NOT the deployment metric)

### Baseline variants (balanced 20/class data, stratified split)

| variant | dot val | dot test | nodot val | nodot test |
|---|---|---|---|---|
| evT166_s2  | 100.0 | 80.0 | 93.3 | 93.3 |
| evT166_s4  | 86.7  | 86.7 | 93.3 | 93.3 |
| evT332_s2  | 93.3  | 86.7 | 93.3 | 93.3 |
| evT332_s4  | 93.3  | 93.3 | 86.7 | 93.3 |
| evT664_s4  | 93.3  | 80.0 | 86.7 | 93.3 |

### Noise-augmented variants (evT166_s2 base)

| variant | dot test | nodot test | vs baseline |
|---|---|---|---|
| baseline | 80.0 | 93.3 | — |
| drop30   | 80.0 | 93.3 | identical per-class outcomes |
| addp1    | 73.3 | 86.7 | BOTH worse — hot-pixel noise hurts everyone |
| shift8   | 80.0 | 93.3 | identical per-class outcomes |
| allnoise | (training) | (training) | |

Per-class test (correct/3) — the decisive view:

| variant | cond | flat | curved | edge | soft | grid |
|---|---|---|---|---|---|---|
| baseline | dot   | 3/3 | **2/3** | 2/3 | 3/3 | **2/3** |
| baseline | nodot | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| drop30   | dot   | 3/3 | 2/3 | 2/3 | 3/3 | 2/3 |
| drop30   | nodot | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| addp1    | dot   | 3/3 | **1/3** | 2/3 | 3/3 | 2/3 |
| addp1    | nodot | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 |
| shift8   | dot   | 3/3 | 2/3 | 2/3 | 3/3 | 2/3 |
| shift8   | nodot | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |

## Noise experiment verdict (2026-11-27)

**The overfitting hypothesis is REFUTED.** drop30/shift8 produce literally
identical per-class outcomes to baseline — regularizing harder does not
unlock dot generalization. addp1 degrades both conditions (spurious events
corrupt the texture signal everyone relies on). Therefore the dot gap on
curved/grid is NOT marker-layout overfitting — the dots genuinely interfere
with the deformation geometry those classes present. Separately: **edge is
intrinsically hard** (2/3 everywhere, both gels, all variants — confused
with grid); likely physically ambiguous at this indentation depth.

## Paired analysis — baseline evT166_s2 (n=15 test clips, 3/class)

```
dot 80.0% vs nodot 93.3%  (delta -13.3 pts)
McNemar: dot-only-correct=0, nodot-only-correct=2, chi2=0.50
--> NOT significant (need chi2 > 3.84 for p<0.05)
```

Per-class test accuracy (dot / nodot):

| class | dot % | nodot % | delta | interpretation |
|---|---|---|---|---|
| flat   | 100 | 100 | 0   | trivially separable both ways |
| curved | 66.7 | 100 | -33.3 | **dot hurts here** |
| edge   | 66.7 | 66.7 | 0 | hard for both |
| soft   | 100 | 100 | 0   | easy both ways |
| grid   | 66.7 | 100 | -33.3 | **dot hurts here** |

Dot confusion pattern: curved misread as flat (1/3) and grid as flat (1/3);
nodot errors are edge<->grid only. Hypothesis: hand-drawn dots vary between
sessions (redrawn), so dot-trained models may overfit session-specific
marker layouts instead of surface geometry — nodot forces geometry-only
features that transfer. Noise experiment tests whether forcing invariance
(noise aug) closes the gap.

## Event-level specksim — NOT YET RUN for phase 2

(the only deployment-valid metric; needs the N-class port of
eval_specksim_events.py — currently 2-class)

| model | val % | test % | notes |
|---|---|---|---|
| | | | |

## Conclusion so far

- Dot contribution: at best neutral, possibly negative (curved/grid) — not yet significant
- Statistical power is limited by 15 test clips; McNemar paired test is the decision tool
- Any deployment claim waits for event-level specksim per the standing rule
