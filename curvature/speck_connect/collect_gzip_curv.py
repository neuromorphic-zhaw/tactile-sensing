import samna
import numpy as np
import h5py
import time
import sys
import os
import tkinter as tk
from datetime import datetime
from threading import Thread

# --- 1. GUI CATEGORY SELECTOR ---
def get_category_via_gui():
    selected = {"type": None}
    root = tk.Tk()
    root.title("Curvature Experiment")
    root.geometry("300x180")
    def set_val(val):
        selected["type"] = val
        root.destroy()
    tk.Label(root, text="Select Surface Type:", font=("Arial", 11, "bold")).pack(pady=15)
    tk.Button(root, text="FLAT", width=15, height=2, command=lambda: set_val("flat"), bg="#e1f5fe").pack(pady=5)
    tk.Button(root, text="CURVED", width=15, height=2, command=lambda: set_val("curved"), bg="#ffebee").pack(pady=5)
    root.mainloop()
    return selected["type"]

# --- 2. SAMNA HELPERS ---
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

# --- 3. MAIN RECORDING LOGIC ---
def run_experiment(duration=10):
    category = get_category_via_gui()
    if not category: return

    root_folder = "dataset_curv"
    save_path = os.path.join(root_folder, category)
    os.makedirs(save_path, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    i = 1
    while True:
        full_path = os.path.join(save_path, f"{date_str}_{category}_{i:02d}.h5")
        if not os.path.exists(full_path): break
        i += 1

    devices = [d for d in samna.device.get_unopened_devices() if d.device_type_name.startswith("Speck2f")]
    if not devices: raise Exception("Speck2f board not found.")
    
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

    print(f"\n CATEGORY: {category.upper()}")
    print(f" SAVING TO: {full_path}")
    input(">>> Press [ENTER] to record... <<<")

    # --- Collection Loop with Live Event Generation Counter ---
    all_dvs, all_pwr = [], []
    dvs_sink.get_events() # Flush
    start_time = time.time()
    
    print(" RECORDING...")
    while time.time() - start_time < duration:
        d_evts = dvs_sink.get_events()
        if d_evts: all_dvs.extend(d_evts)
        p_evts = power_sink.get_events()
        if p_evts: all_pwr.extend(p_evts)
        
        # Live Terminal GUI update
        elapsed = time.time() - start_time
        print(f"\r  Time: {elapsed:.1f}/10s | Events Generated: {len(all_dvs):,}", end="", flush=True)
        time.sleep(0.05)

    # --- Save Data with GZIP Compression ---
    print(f"\n\nSaving {len(all_dvs):,} events to HDF5 with GZIP...")
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
        f.attrs["event_count"] = len(all_dvs)

    dk.get_power_monitor().stop_auto_power_measurement()
    graph.stop()
    print("✅ Done. Close visualizer window to exit.")
    gui_thread.join()

if __name__ == "__main__":
    run_experiment(duration=10)
