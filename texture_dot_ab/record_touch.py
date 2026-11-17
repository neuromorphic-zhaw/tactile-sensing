"""record_touch.py — guided tactile recorder with phase tagging (dot A/B).

One GUI drives the whole session:
  1. SETUP   — pick gel (DOT / NODOT) and surface class (flat/curved/edge/…)
  2. RECORD  — live event count + rate; click APPROACH / TOUCH / RETREAT as
               you perform each motion phase; phase marks are timestamped and
               saved into the .h5 (phase_t / phase_name) alongside x,y,t,p
               and power traces — so approach-only vs touch-only training
               slices are available later
  3. Loop    — ENTER/next clip, change class/gel, or quit

Hardware mode talks to the devkit exactly like collect_gzip_curv.py
(dvs_filter + hot-pixel filter on). --mock runs the identical GUI/save path
with a synthetic event source (phase-dependent rates) — no devkit needed.

    python record_touch.py --mock
    python record_touch.py --duration 12 --root ../data/dataset_tex
"""

import argparse
import os
import sys
import threading
import time
import tkinter as tk
from datetime import datetime

import h5py
import numpy as np

PHASES = ["approach", "touch", "retreat"]
PHASE_COLORS = {"idle": "#9e9e9e", "approach": "#ffb300", "touch": "#e53935",
                "retreat": "#1e88e5"}

# mock event rates (events/sec) per phase — realistic touch dynamics
MOCK_RATES = {"idle": 300, "approach": 8000, "touch": 22000, "retreat": 5000}


class EventSource:
    """Uniform event/power source. Mock synthesizes; hardware wraps samna."""

    def __init__(self, mock):
        self.mock = mock
        self.dvs, self.pwr = [], []
        self.phase = "idle"
        self._stop = threading.Event()
        self._thread = None
        self.dk = self.graph = None
        self.dvs_sink = self.pwr_sink = None

    # ---------- hardware ----------
    def open_hardware(self, viz=True):
        import samna
        devices = [d for d in samna.device.get_unopened_devices()
                   if d.device_type_name.startswith("Speck2f")]
        if not devices:
            raise RuntimeError("Speck2f board not found (use --mock to test)")
        self.dk = samna.device.open_device(devices[0])
        time.sleep(0.5)
        self.graph = samna.graph.EventFilterGraph()
        if viz:
            self._open_visualizer(samna, "tcp://0.0.0.0:40001")
        self.dvs_sink = samna.graph.sink_from(self.dk.get_model_source_node())
        self.pwr_sink = samna.graph.sink_from(
            self.dk.get_power_monitor().get_source_node())
        cfg = samna.speck2f.configuration.SpeckConfiguration()
        cfg.dvs_layer.monitor_enable = True
        cfg.dvs_layer.raw_monitor_enable = True
        cfg.dvs_filter.enable = True
        cfg.dvs_filter.hot_pixel_filter_enable = True
        cfg.dvs_filter.threshold = 10
        self.dk.get_model().apply_configuration(cfg)
        self.dk.get_stop_watch().start()
        self.dk.get_power_monitor().start_auto_power_measurement(100.0)
        self.graph.start()

    def _open_visualizer(self, samna, endpoint):
        gui_cmd = (f"import samna, samnagui; "
                   f"samnagui.run_visualizer('{endpoint}', 0.75, 0.75)")
        threading.Thread(target=os.system,
                         args=(f'{sys.executable} -c "{gui_cmd}"',),
                         daemon=True).start()
        _, _, streamer = self.graph.sequential(
            [self.dk.get_model_source_node(), "Speck2fDvsToVizConverter",
             "VizEventStreamer"])
        self.graph.sequential(
            [self.dk.get_power_monitor().get_source_node(),
             "MeasurementToVizConverter", streamer])
        cfg_src, _ = self.graph.sequential(
            [samna.BasicSourceNode_ui_event(), streamer])
        streamer.set_streamer_endpoint(endpoint)
        while streamer.wait_for_receiver_count() == 0:
            time.sleep(0.1)
        cfg_src.write([samna.ui.VisualizerConfiguration(plots=[
            samna.ui.ActivityPlotConfiguration(
                128, 128, "DVS Layer", layout=[0, 0, 1, 0.7]),
            samna.ui.PowerMeasurementPlotConfiguration(
                title="Power Consumption", channel_count=5,
                line_names=["io", "ram", "logic", "vddd", "vdda"],
                layout=[0, 0.7, 1, 1], show_x_span=10, label_interval=2,
                max_y_rate=1.5, show_point_circle=False, default_y_max=1,
                y_label_name="power (mW)")])])

    # ---------- acquisition thread ----------
    def start(self):
        self.dvs, self.pwr = [], []
        self._stop.clear()
        if self.mock:
            self._thread = threading.Thread(target=self._mock_loop, daemon=True)
        else:
            self._thread = threading.Thread(target=self._hw_loop, daemon=True)
            self.dvs_sink.get_events()  # flush stale
        self._thread.start()

    def _mock_loop(self):
        t = 0
        rng = np.random.default_rng()
        while not self._stop.is_set():
            rate = MOCK_RATES.get(self.phase, 300)
            n = int(rate * 0.05)  # 50 ms tick
            if n:
                xs = rng.integers(0, 128, n)
                ys = rng.integers(0, 128, n)
                ps = rng.integers(0, 2, n)
                ts = t + np.sort(rng.integers(0, 50_000, n))
                self.dvs.extend(zip(xs.tolist(), ys.tolist(),
                                    ts.tolist(), ps.tolist()))
            t += 50_000
            self.pwr.append((t, rng.uniform(0.9, 1.1, 5).tolist()))
            time.sleep(0.05)

    def _hw_loop(self):
        while not self._stop.is_set():
            d = self.dvs_sink.get_events()
            if d:
                self.dvs.extend((e.x, e.y, e.timestamp,
                                 getattr(e, "p", getattr(e, "feature", 1)))
                                for e in d)
            p = self.pwr_sink.get_events()
            if p:
                self.pwr.extend((e.timestamp,
                                 getattr(e, "value",
                                         getattr(e, "point", []))) for e in p)
            time.sleep(0.02)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def close(self):
        if self.dk is not None:
            try:
                self.dk.get_power_monitor().stop_auto_power_measurement()
                self.graph.stop()
            except Exception:
                pass


def save_clip(path, dvs, pwr, phases, category, condition):
    """dvs: list of (x,y,t,p). phases: list of (t_offset_sec, name)."""
    arr = np.asarray(dvs, dtype=np.int64) if dvs else np.zeros((0, 4), np.int64)
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=arr[:, 0], compression="gzip")
        f.create_dataset("y", data=arr[:, 1], compression="gzip")
        f.create_dataset("t", data=arr[:, 2], compression="gzip")
        f.create_dataset("p", data=arr[:, 3], compression="gzip")
        if pwr:
            f.create_dataset("power_t", data=[p[0] for p in pwr],
                             compression="gzip")
            try:
                f.create_dataset("power_v", data=[p[1] for p in pwr],
                                 compression="gzip")
                f.attrs["power_labels"] = ["io", "ram", "logic", "vddd", "vdda"]
            except Exception:
                pass
        if phases:
            f.create_dataset("phase_t", data=[ph[0] for ph in phases])
            f.create_dataset("phase_name",
                             data=np.array([ph[1] for ph in phases],
                                           dtype=h5py.string_dtype()))
        f.attrs["label"] = category
        f.attrs["condition"] = condition
        f.attrs["event_count"] = len(dvs)


class RecorderGUI:
    def __init__(self, args):
        self.args = args
        self.src = EventSource(args.mock)
        self.recording = False
        self.t0 = 0.0
        self.last_n = 0
        self.last_t = 0.0
        self.phases = []
        self.condition = None
        self.category = None
        self.clip_no = 1

        self.root = tk.Tk()
        self.root.title("Speck-touch recorder — texture_dot_ab")
        self.root.geometry("460x480")
        self.frame = tk.Frame(self.root)
        self.frame.pack(expand=True, fill="both")
        self._setup_screen()

    # ---------- screens ----------
    def _clear(self):
        for w in self.frame.winfo_children():
            w.destroy()

    def _setup_screen(self):
        self._clear()
        tk.Label(self.frame, text="SESSION SETUP",
                 font=("Arial", 14, "bold")).pack(pady=10)

        tk.Label(self.frame, text="Gel condition:",
                 font=("Arial", 11, "bold")).pack()
        gel_row = tk.Frame(self.frame); gel_row.pack(pady=4)
        for cond, col in [("dot", "#fff3e0"), ("nodot", "#e1f5fe")]:
            tk.Button(gel_row, text=cond.upper(), width=12, height=2, bg=col,
                      command=lambda c=cond: self._pick_gel(c)).pack(
                      side="left", padx=6)
        self.gel_lbl = tk.Label(self.frame, text="(none)", fg="#666")
        self.gel_lbl.pack()

        tk.Label(self.frame, text="Surface class:",
                 font=("Arial", 11, "bold")).pack(pady=(14, 2))
        classes = [c.strip() for c in self.args.classes.split(",") if c.strip()]
        self.cls_btns = []
        for cls in classes:
            b = tk.Button(self.frame, text=cls.upper(), width=16, height=1,
                          command=lambda c=cls: self._pick_class(c))
            b.pack(pady=2); self.cls_btns.append((b, cls))
        self.cls_lbl = tk.Label(self.frame, text="(none)", fg="#666")
        self.cls_lbl.pack(pady=2)

        self.start_btn = tk.Button(self.frame, text="OPEN RECORDER",
                                   width=20, height=2, bg="#c8e6c9",
                                   state="disabled",
                                   command=self._record_screen)
        self.start_btn.pack(pady=14)

    def _pick_gel(self, c):
        self.condition = c
        self.gel_lbl.config(text=f"gel: {c}", fg="#000")
        self._maybe_enable()

    def _pick_class(self, c):
        self.category = c
        self.cls_lbl.config(text=f"class: {c}", fg="#000")
        for b, name in self.cls_btns:
            b.config(relief="sunken" if name == c else "raised")
        self._maybe_enable()

    def _maybe_enable(self):
        if self.condition and self.category:
            self.start_btn.config(state="normal")

    def _record_screen(self):
        self._clear()
        hdr = (f"gel: {self.condition}   class: {self.category}   "
               f"clip: {self.clip_no:02d}")
        tk.Label(self.frame, text=hdr, font=("Arial", 12, "bold")).pack(pady=8)

        self.count_lbl = tk.Label(self.frame, text="0",
                                  font=("Courier", 36, "bold"))
        self.count_lbl.pack()
        self.rate_lbl = tk.Label(self.frame, text="0 events/s",
                                 font=("Courier", 14), fg="#555")
        self.rate_lbl.pack(pady=(0, 8))

        self.phase_lbl = tk.Label(self.frame, text="PHASE: idle",
                                  font=("Arial", 13, "bold"),
                                  fg="white", bg=PHASE_COLORS["idle"],
                                  width=22)
        self.phase_lbl.pack(pady=6)

        row = tk.Frame(self.frame); row.pack(pady=4)
        for ph in PHASES:
            tk.Button(row, text=ph.upper(), width=10, height=2,
                      bg=PHASE_COLORS[ph], fg="white",
                      command=lambda p=ph: self._mark(p)).pack(
                      side="left", padx=4)

        self.rec_btn = tk.Button(self.frame, text="● START RECORDING",
                                 width=22, height=2, bg="#ef9a9a",
                                 command=self._toggle_rec)
        self.rec_btn.pack(pady=12)
        self.status_lbl = tk.Label(self.frame, text="ready", fg="#666")
        self.status_lbl.pack()

        back = tk.Frame(self.frame); back.pack(pady=10)
        tk.Button(back, text="NEXT CLIP →", width=14,
                  command=self._next_clip).pack(side="left", padx=4)
        tk.Button(back, text="CHANGE SETUP", width=14,
                  command=self._setup_screen).pack(side="left", padx=4)
        tk.Button(back, text="QUIT", width=8,
                  command=self.root.destroy).pack(side="left", padx=4)

    # ---------- recording ----------
    def _mark(self, phase):
        if not self.recording:
            self.src.phase = phase  # preview rate in mock
            self.phase_lbl.config(text=f"PHASE: {phase}",
                                  bg=PHASE_COLORS[phase])
            return
        self.src.phase = phase
        self.phases.append((time.time() - self.t0, phase))
        self.phase_lbl.config(text=f"PHASE: {phase}",
                              bg=PHASE_COLORS[phase])

    def _toggle_rec(self):
        if not self.recording:
            self.recording = True
            self.phases = []
            self.src.start()
            self.t0 = time.time()
            self.last_n, self.last_t = 0, self.t0
            self.src.phase = "idle"
            self.rec_btn.config(text="■ STOP + SAVE", bg="#a5d6a7")
            self.status_lbl.config(text="recording…")
            self._tick()
        else:
            self.recording = False
            self.src.stop()
            self._save()
            self.rec_btn.config(text="● START RECORDING", bg="#ef9a9a")

    def _tick(self):
        if not self.recording:
            return
        n = len(self.src.dvs)
        now = time.time()
        dt = now - self.last_t
        rate = (n - self.last_n) / dt if dt > 0 else 0
        self.last_n, self.last_t = n, now
        self.count_lbl.config(text=f"{n:,}")
        self.rate_lbl.config(text=f"{rate:,.0f} events/s   "
                                  f"{now - self.t0:4.1f}s")
        if self.args.duration and now - self.t0 >= self.args.duration:
            self._toggle_rec()
            return
        self._after = self.root.after(100, self._tick)

    def _save(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(self.args.root, self.condition, self.category)
        os.makedirs(save_dir, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(
                save_dir, f"{date_str}_{self.category}_{i:02d}.h5")):
            i += 1
        path = os.path.join(save_dir,
                            f"{date_str}_{self.category}_{i:02d}.h5")
        save_clip(path, self.src.dvs, self.src.pwr, self.phases,
                  self.category, self.condition)
        ph_str = " → ".join(p for _, p in self.phases) or "none"
        self.status_lbl.config(
            text=f"saved {os.path.basename(path)}  "
                 f"({len(self.src.dvs):,} ev | phases: {ph_str})")
        self.clip_no += 1

    def _next_clip(self):
        self._record_screen()

    def run(self):
        if not self.args.mock:
            self.src.open_hardware()
        self.root.mainloop()
        self.src.stop()
        self.src.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="no devkit — synthetic events, full GUI path")
    ap.add_argument("--classes", default="flat,curved,edge")
    ap.add_argument("--duration", type=float, default=0,
                    help="auto-stop after N sec (0 = manual stop)")
    ap.add_argument("--root", default="../data/dataset_tex")
    RecorderGUI(ap.parse_args()).run()
