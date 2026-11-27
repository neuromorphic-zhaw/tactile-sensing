# force — stage 3 (planned)

Third thesis stage: **force-sensor integration** — fuse the Speck2f
event-stream with a force/torque channel to understand applied force
during touch.

Planned layout (same conventions as stage 1/2):

```
data/dataset_force/<class>/<date>_<class>_<nn>.h5   raw clips, force
                                                   channel stored as a
                                                   parallel dataset in
                                                   the same .h5 (or a
                                                   sidecar .csv/.npz)
data/dataset_force_evT166/<class>/*.npz             derived rasterization
force/runs/                                       trainer output
```

TODO once hardware is settled:
- decide the force channel source + sampling rate, and how it is logged
  alongside the event stream (h5 dataset vs sidecar, timestamp alignment)
- fork the stage-2 recorder (`phase2_texture_dot_ab/record_touch.py`) — its
  phase-tagging state machine already handles synchronized aux channels
- reuse `prep_count_bins.py` rasterization; extend to carry the force
  channel through to the trainer
