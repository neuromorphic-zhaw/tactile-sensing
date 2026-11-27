# devin_rawevent — raw event-stream models (all dead ends)

Feed events to the SNN without rasterization parity tricks — closest to
ANN-event literature. Variants: `ev_base` (parity anchor), `ev_pol`
(polarity), `ev_augs`, `ev_warp` (time-warp aug), `ev_bin5ms`,
`ev_mixres` (mixed resolution).

**Verdict: abandon.** Event-level specksim results were 36.67–46.67%
val where they ran at all — `ev_base` and `evT332_s4` deadlock the C++
sim outright (burned 8h with ~1min CPU), `ev_bin5ms`/`ev_mixres`
bad_alloc. Kept for provenance only; use count-binning
(`../devin_evcount/`) for any new work.
