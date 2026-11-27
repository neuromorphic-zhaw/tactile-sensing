# devin_ssm — SNN + S4D state-space hybrids

The 9-core SNN runs on-chip; a small S4D SSM head runs host-side as a
learned replacement for the windowed majority vote. Naming:
`ssm_<backbone>[_frz|_l8|_l8ft]`:

- `ssm_evT166` / `ssm_t10ms` — count-bin vs 10ms time-bin data
- `_frz` = frozen backbone, SSM trained on readout spikes (T',2)
- `_l8`  = SSM trained on 64-dim last-conv feature spikes (SNN as pure
  feature extractor; `eval_specksim_ssm.py` auto-detects via ckpt width)
- `_l8ft` = l8 features + joint backbone fine-tune (deprioritized)
- no suffix = joint training from scratch

**Verdict so far**: all ≤63.33% event-level val, `ssm_evT166_frz` went
degenerate all-flat on test (46.67%). wclamp-backbone variants are
unrunnable in specksim (bad_alloc at every event cap). The windowed
majority vote remains unbeaten — see `../devin_evcount/`.
