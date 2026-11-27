"""phase2_texture_dot_ab collector — dot vs markerless gel A/B.

Adapted from phase1_curvature/speck_connect/collect_gzip_curv.py. Differences:
  - class list is arbitrary (--classes); gel condition is a CLI flag
    (--condition dot|nodot) so the same surface set can be replayed under
    both gels with identical numbering
  - saves under data/phase2_texture_dot_ab/dataset_tex_raw/<condition>/<class>/
  - records N clips in one session without restarting (ENTER = next clip,
    'c' = change class, 'q' = quit) so a paired protocol stays in order

Raw clip format identical to dataset_curv: x/y/t/p event arrays + power_t/
power_v traces + label/condition attrs, gzip-compressed .h5.
"""

import argparse
import os
import sys
import time
import tkinter as tk
from datetime import datetime
from threading import Thread

import h5py
import numpy as np
import samna


def get_category_via_gui(classes, condition):
    selected = {"type": None}
    root = tk.Tk()
    root.title(f"Texture Experiment — gel: {condition}")
    root.geometry("320x420")
    def set_val(val):
        selected["type"] = val
        root.destroy()
    tk.Label(root, text=f"Gel: {condition}\nSelect Surface Class:",
             font=("Arial", 11, "bold")).pack(pady=15)
    colors = ["#e1f5fe", "#ffebee", "#e8f5e9", "#fff3e0", "#f3e5f5", "#ede7f6"]
    for i, cls in enumerate(classes):
        tk.Button(root, text=cls.upper(), width=15, height=2,
                  command=lambda v=cls: set_val(v),
                  bg=colors[i % len(colors)]).pack(pady=4)
    root.mainloop()
    return selected["type"]


def open_visualizer(width_proportion, height_proportion, receiver_endpoint):
    gui_cmd = f"import samna, samnagui; samnagui.run_visualizer('{receiver_endpoint}', {width_proportion}, {height_proportion})"
    os_cmd = f'{sys.executable} -c "{gui_cmd}"'
    gui_thread = Thread(target=os.system, args=(os_cmd,))
    gui_thread.start()
    return gui_thread


def build_samna_event_route(dk, graph, endpoint):
    _, _, streamer = graph.sequential(
        [dk.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"]
    )
    graph.sequential(
        [dk.get_power_monitor().get_source_node(), "MeasurementToVizConverter", streamer]
    )
    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])
    streamer.set_streamer_endpoint(endpoint)

    while streamer.wait_for_receiver_count() == 0:
        time.sleep(0.1)

    visualizer_config = samna.ui.VisualizerConfiguration(
        plots=[
            samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer", layout=[0, 0, 1, 0.7]),
            samna.ui.PowerMeasurementPlotConfiguration(
                title="Power Consumption",
                channel_count=5,
                line_names=["io", "ram", "logic", "vddd", "vdda"],
                layout=[0, 0.7, 1, 1],
                show_x_span=10,
                label_interval=2,
                max_y_rate=1.5,
                show_point_circle=False,
                default_y_max=1,
                y_label_name="power (mW)",
            )
        ]
    )
    config_source.write([visualizer_config])


def next_clip_path(root_folder, condition, category, date_str):
    save_path = os.path.join(root_folder, condition, category)
    os.makedirs(save_path, exist_ok=True)
    i = 1
    while True:
        full_path = os.path.join(save_path, f"{date_str}_{category}_{i:02d}.h5")
        if not os.path.exists(full_path):
            return full_path
        i += 1


def record_clip(dvs_sink, power_sink, duration):
    all_dvs, all_pwr = [], []
    dvs_sink.get_events()
    start_time = time.time()
    print(" RECORDING...")
    while time.time() - start_time < duration:
        d_evts = dvs_sink.get_events()
        if d_evts: all_dvs.extend(d_evts)
        p_evts = power_sink.get_events()
        if p_evts: all_pwr.extend(p_evts)
        elapsed = time.time() - start_time
        print(f"\r  Time: {elapsed:.1f}/{duration}s | Events: {len(all_dvs):,}",
              end="", flush=True)
        time.sleep(0.05)
    print()
    return all_dvs, all_pwr


def save_clip(full_path, all_dvs, all_pwr, category, condition):
    print(f"Saving {len(all_dvs):,} events to HDF5 with GZIP...")
    with h5py.File(full_path, "w") as f:
        if all_dvs:
            f.create_dataset("x", data=[e.x for e in all_dvs], compression="gzip")
            f.create_dataset("y", data=[e.y for e in all_dvs], compression="gzip")
            f.create_dataset("t", data=[e.timestamp for e in all_dvs], compression="gzip")
            p_data = [getattr(e, 'p', getattr(e, 'feature', 1)) for e in all_dvs]
            f.create_dataset("p", data=p_data, compression="gzip")
        if all_pwr:
            power_values = [getattr(p, 'value', getattr(p, 'point', [])) for p in all_pwr]
            f.create_dataset("power_v", data=power_values, compression="gzip")
            f.create_dataset("power_t", data=[p.timestamp for p in all_pwr], compression="gzip")
            f.attrs["power_labels"] = ["io", "ram", "logic", "vddd", "vdda"]
        f.attrs["label"] = category
        f.attrs["condition"] = condition
        f.attrs["event_count"] = len(all_dvs)


def run_experiment(args):
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    root_folder = args.root
    date_str = datetime.now().strftime("%Y-%m-%d")

    devices = [d for d in samna.device.get_unopened_devices()
               if d.device_type_name.startswith("Speck2f")]
    if not devices:
        raise Exception("Speck2f board not found.")
    dk = samna.device.open_device(devices[0])
    time.sleep(0.5)

    endpoint = "tcp://0.0.0.0:40001"
    gui_thread = open_visualizer(0.75, 0.75, endpoint)
    graph = samna.graph.EventFilterGraph()
    build_samna_event_route(dk, graph, endpoint)

    dvs_sink = samna.graph.sink_from(dk.get_model_source_node())
    power_sink = samna.graph.sink_from(dk.get_power_monitor().get_source_node())
    graph.start()

    config = samna.speck2f.configuration.SpeckConfiguration()
    config.dvs_layer.monitor_enable = True
    config.dvs_layer.raw_monitor_enable = True
    config.dvs_filter.enable = True
    config.dvs_filter.hot_pixel_filter_enable = True
    config.dvs_filter.threshold = 10
    dk.get_model().apply_configuration(config)

    dk.get_stop_watch().start()
    dk.get_power_monitor().start_auto_power_measurement(100.0)

    category = get_category_via_gui(classes, args.condition)
    if not category:
        return
    print(f"\n CONDITION: {args.condition} | CLASS: {category}")
    print(" Keep the SAME surface order when you re-run under the other gel.")

    while True:
        full_path = next_clip_path(root_folder, args.condition, category, date_str)
        print(f"\n NEXT CLIP: {os.path.basename(full_path)}")
        key = input(">>> [ENTER]=record  c=change class  q=quit <<< ").strip().lower()
        if key == "q":
            break
        if key == "c":
            category = get_category_via_gui(classes, args.condition)
            if not category:
                break
            print(f" CLASS -> {category}")
            continue

        all_dvs, all_pwr = record_clip(dvs_sink, power_sink, args.duration)
        save_clip(full_path, all_dvs, all_pwr, category, args.condition)

    dk.get_power_monitor().stop_auto_power_measurement()
    graph.stop()
    print("Done. Close visualizer window to exit.")
    gui_thread.join()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["dot", "nodot"], required=True)
    ap.add_argument("--classes", default="flat,curved,edge,soft,grid",
                    help="comma-separated class list (must match across conditions)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--root", default="../data/phase2_texture_dot_ab/dataset_tex_raw")
    run_experiment(ap.parse_args())
