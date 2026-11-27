"""record_touch.py — guided tactile recorder with phase tagging (dot A/B).

Spacebar-driven recording — the whole clip is 4 presses:

    READY    --SPACE-->  start recording, phase = approach   (t=0 mark)
    APPROACH --SPACE-->  phase = touch; precise 10.00 s contact timer starts
    TOUCH    --SPACE-->  phase = retreat
    RETREAT  --SPACE-->  stop recording + save .h5, back to READY

During TOUCH a countdown shows "10.00 s" (perf_counter-based, not tick-
counted). At the deadline a "touch_10s" mark is written + bell; press
SPACE when it ends to move to RETREAT. Pressing SPACE early is allowed
but flagged with the actual touch duration.

Every phase mark stores (t_sec, name, event_index) — phase_idx aligns
marks exactly to the x/y/t/p stream; .h5 also gets touch_duration_s.

Hardware mode talks to the devkit exactly like collect_gzip_curv.py
(dvs_filter + hot-pixel filter on). --mock runs the identical GUI/save
path with a synthetic event source (phase-dependent rates) — no devkit
needed.

    python record_touch.py --mock
    python record_touch.py --root ../data/phase2_texture_dot_ab/dataset_tex
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
TOUCH_SECONDS = 10.0  # precise contact timer

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
        # bounded wait — a dead samnagui must never block the recorder;
        # the GUI connects fine even if it attaches late
        for _ in range(100):
            if streamer.wait_for_receiver_count() != 0:
                break
            time.sleep(0.1)
        else:
            print("warn: samnagui did not connect in 10 s — continuing without viz")
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


def save_clip(path, dvs, pwr, phases, category, condition, touch_duration=None):
    """dvs: list of (x,y,t,p). phases: list of (t_sec, name, event_index)."""
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
            f.create_dataset("phase_idx", data=[ph[2] for ph in phases])
        f.attrs["label"] = category
        f.attrs["condition"] = condition
        f.attrs["event_count"] = len(dvs)
        if touch_duration is not None:
            f.attrs["touch_duration_s"] = touch_duration


class RecorderGUI:
    def __init__(self, args):
        self.args = args
        self.src = EventSource(args.mock)
        self.state = "setup"  # setup|ready|approach|touch|retreat
        self.recording = False
        self.t0 = 0.0
        self.last_n = 0
        self.last_t = 0.0
        self.phases = []           # (t_sec, name, event_idx)
        self.touch_t0 = None       # perf_counter at touch start
        self.touch_deadline = None # perf_counter deadline = t0 + 10 s
        self.touch_duration = None
        self._timer_after = None
        self._tick_after = None
        self.condition = None
        self.category = None
        self.clip_no = 1

        self.root = tk.Tk()
        self.root.title("Speck-touch recorder — phase2_texture_dot_ab")
        self.root.geometry("480x540")
        self.frame = tk.Frame(self.root)
        self.frame.pack(expand=True, fill="both")
        self.root.bind("<space>", self._space)
        self._setup_screen()

    # ---------- screens ----------
    def _clear(self):
        for w in self.frame.winfo_children():
            w.destroy()

    def _btn(self, parent, **kw):
        kw.setdefault("takefocus", 0)   # space must never click a button
        return tk.Button(parent, **kw)

    def _setup_screen(self):
        self.state = "setup"
        self._clear()
        tk.Label(self.frame, text="SESSION SETUP",
                 font=("Arial", 14, "bold")).pack(pady=10)

        tk.Label(self.frame, text="Gel condition:",
                 font=("Arial", 11, "bold")).pack()
        gel_row = tk.Frame(self.frame); gel_row.pack(pady=4)
        for cond, col in [("dot", "#fff3e0"), ("nodot", "#e1f5fe")]:
            self._btn(gel_row, text=cond.upper(), width=12, height=2, bg=col,
                      command=lambda c=cond: self._pick_gel(c)).pack(
                      side="left", padx=6)
        self.gel_lbl = tk.Label(self.frame, text="(none)", fg="#666")
        self.gel_lbl.pack()

        tk.Label(self.frame, text="Surface class:",
                 font=("Arial", 11, "bold")).pack(pady=(14, 2))
        classes = [c.strip() for c in self.args.classes.split(",") if c.strip()]
        self.cls_btns = []
        for cls in classes:
            b = self._btn(self.frame, text=cls.upper(), width=16, height=1,
                          command=lambda c=cls: self._pick_class(c))
            b.pack(pady=2); self.cls_btns.append((b, cls))
        self.cls_lbl = tk.Label(self.frame, text="(none)", fg="#666")
        self.cls_lbl.pack(pady=2)

        self.start_btn = self._btn(self.frame, text="OPEN RECORDER",
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
        self.state = "ready"
        self._clear()
        hdr = (f"gel: {self.condition}   class: {self.category}   "
               f"clip: {self.clip_no:02d}")
        tk.Label(self.frame, text=hdr, font=("Arial", 12, "bold")).pack(pady=8)

        self.state_lbl = tk.Label(self.frame, text="READY",
                                  font=("Arial", 16, "bold"),
                                  fg="white", bg=PHASE_COLORS["idle"],
                                  width=24)
        self.state_lbl.pack(pady=4)
        self.hint_lbl = tk.Label(self.frame,
                                 text="SPACE = start (approach)",
                                 font=("Arial", 12), fg="#333")
        self.hint_lbl.pack()

        self.count_lbl = tk.Label(self.frame, text="0",
                                  font=("Courier", 32, "bold"))
        self.count_lbl.pack(pady=(8, 0))
        self.rate_lbl = tk.Label(self.frame, text="0 events/s",
                                 font=("Courier", 13), fg="#555")
        self.rate_lbl.pack()

        # precise contact timer (visible during touch)
        self.timer_lbl = tk.Label(self.frame, text="",
                                  font=("Courier", 22, "bold"), fg="#b71c1c")
        self.timer_lbl.pack(pady=4)

        # phase indicators (not clickable — space drives everything)
        row = tk.Frame(self.frame); row.pack(pady=6)
        self.phase_lbls = {}
        for ph in PHASES:
            lb = tk.Label(row, text=ph.upper(), width=10, height=1,
                          bg="#bdbdbd", fg="white", font=("Arial", 10, "bold"))
            lb.pack(side="left", padx=4)
            self.phase_lbls[ph] = lb

        self.status_lbl = tk.Label(self.frame, text="press SPACE to start",
                                   fg="#666", wraplength=440)
        self.status_lbl.pack(pady=6)

        back = tk.Frame(self.frame); back.pack(pady=8)
        self._btn(back, text="CHANGE SETUP", width=14,
                  command=self._setup_screen).pack(side="left", padx=4)
        self._btn(back, text="QUIT", width=8,
                  command=self.root.destroy).pack(side="left", padx=4)

    # ---------- spacebar state machine ----------
    def _space(self, _evt):
        if self.state == "setup":
            return "break"
        if self.state == "ready":
            self._begin()
        elif self.state == "approach":
            self._enter("touch")
            self.touch_t0 = time.perf_counter()
            self.touch_deadline = self.touch_t0 + TOUCH_SECONDS
            self._touch_tick()
        elif self.state == "touch":
            self.touch_duration = time.perf_counter() - self.touch_t0
            if self._timer_after is not None:
                self.root.after_cancel(self._timer_after)
                self._timer_after = None
            self.timer_lbl.config(text="", fg="#b71c1c")
            if self.touch_duration < TOUCH_SECONDS - 0.05:
                self.status_lbl.config(
                    text=f"touch ended early: {self.touch_duration:.2f} s")
            self._enter("retreat")
        elif self.state == "retreat":
            self._stop_save()
        return "break"

    def _begin(self):
        self.recording = True
        self.state = "approach"
        self.phases = []
        self.touch_t0 = self.touch_deadline = None
        self.touch_duration = None
        self.src.start()
        self.t0 = time.perf_counter()
        self.last_n, self.last_t = 0, self.t0
        self.src.phase = "approach"
        self.phases.append((0.0, "approach", len(self.src.dvs)))
        self._light_phase("approach")
        self._tick()

    def _enter(self, phase):
        """Mark a new phase + update banner. phase in {touch, retreat}."""
        self.state = phase
        self.src.phase = phase
        t = time.perf_counter() - self.t0
        self.phases.append((t, phase, len(self.src.dvs)))
        self._light_phase(phase)

    def _light_phase(self, phase):
        """Banner + indicator highlight + next-step hint, all from state."""
        hints = {"approach": "SPACE = surface contact",
                 "touch": f"SPACE = retreat (hold {TOUCH_SECONDS:.0f} s)",
                 "retreat": "SPACE = stop + save"}
        self.state_lbl.config(text=phase.upper(),
                              bg=PHASE_COLORS[phase])
        self.hint_lbl.config(text=hints.get(phase, ""))
        for ph, lb in self.phase_lbls.items():
            lb.config(bg=PHASE_COLORS[ph] if ph == phase else "#bdbdbd")

    def _stop_save(self):
        self.recording = False
        self.src.stop()
        if self._timer_after is not None:
            self.root.after_cancel(self._timer_after)
            self._timer_after = None
        if self._tick_after is not None:
            self.root.after_cancel(self._tick_after)
            self._tick_after = None
        self._save()
        self.state = "ready"
        self.state_lbl.config(text="READY", bg=PHASE_COLORS["idle"])
        self.hint_lbl.config(text="SPACE = start (approach)")
        self.timer_lbl.config(text="")
        for lb in self.phase_lbls.values():
            lb.config(bg="#bdbdbd")

    # ---------- precise 10 s contact timer ----------
    def _touch_tick(self):
        rem = self.touch_deadline - time.perf_counter()
        if rem <= 0:
            # exact intended instant, not the late callback time
            t = self.touch_deadline - self.t0
            self.phases.append((t, "touch_10s", len(self.src.dvs)))
            self._timer_after = None
            self.timer_lbl.config(text="10 s DONE — SPACE = RETREAT",
                                  fg="#1b5e20")
            self.root.bell()
            return
        self.timer_lbl.config(text=f"{rem:6.2f} s")
        self._timer_after = self.root.after(10, self._touch_tick)

    # ---------- live counters ----------
    def _tick(self):
        if not self.recording:
            return
        n = len(self.src.dvs)
        now = time.perf_counter()
        dt = now - self.last_t
        rate = (n - self.last_n) / dt if dt > 0 else 0
        self.last_n, self.last_t = n, now
        self.count_lbl.config(text=f"{n:,}")
        self.rate_lbl.config(text=f"{rate:,.0f} events/s   "
                                  f"{now - self.t0:4.1f}s")
        self._tick_after = self.root.after(100, self._tick)

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
                  self.category, self.condition, self.touch_duration)
        ph_str = " → ".join(p for _, p, _ in self.phases) or "none"
        td = (f" | touch {self.touch_duration:.2f}s"
              if self.touch_duration is not None else "")
        self.status_lbl.config(
            text=f"saved {os.path.basename(path)}  "
                 f"({len(self.src.dvs):,} ev | phases: {ph_str}{td})")
        self.clip_no += 1
        # update header for the next clip without rebuilding the screen
        self._record_screen_header()

    def _record_screen_header(self):
        hdr = (f"gel: {self.condition}   class: {self.category}   "
               f"clip: {self.clip_no:02d}")
        for w in self.frame.winfo_children():
            if isinstance(w, tk.Label) and w.cget("text").startswith("gel:"):
                w.config(text=hdr)
                break

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
    ap.add_argument("--classes", default="flat,curved,edge,soft,grid")
    ap.add_argument("--root", default="../data/phase2_texture_dot_ab/dataset_tex")
    RecorderGUI(ap.parse_args()).run()
