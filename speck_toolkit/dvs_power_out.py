import samna
import sys
import os
import time
from threading import Thread

def open_speck2f_dev_kit():
    devices = [
        device
        for device in samna.device.get_unopened_devices()
        if device.device_type_name.startswith("Speck2f")
    ]
    assert devices, "Speck2f board not found"

    return samna.device.open_device(devices[0])

def build_samna_event_route(dk, graph, endpoint):
    # build a graph in samna to show dvs
    _, _, streamer = graph.sequential(
        [dk.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"]
    )

    # --- NEW: Route the power measurements to the visualizer ---
    graph.sequential(
        [dk.get_power_monitor().get_source_node(), "MeasurementToVizConverter", streamer]
    )
    # -----------------------------------------------------------

    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])

    streamer.set_streamer_endpoint(endpoint)
    if streamer.wait_for_receiver_count() == 0:
        raise Exception(f'connecting to visualizer on {endpoint} fails')

    # --- UPDATED: Configures both the DVS window and the Power graph ---
    visualizer_config = samna.ui.VisualizerConfiguration(
        plots=[
            samna.ui.ActivityPlotConfiguration(
                128, 128, "DVS Layer", layout=[0, 0, 1, 0.7] # Top 70% of the window
            ),
            samna.ui.PowerMeasurementPlotConfiguration(
                title="Power Consumption",
                channel_count=5, # Speck2f has 5 power tracks
                line_names=["io", "ram", "logic", "vddd", "vdda"],
                layout=[0, 0.7, 1, 1], # Bottom 30% of the window
                show_x_span=10,
                label_interval=2,
                max_y_rate=1.5,
                show_point_circle=False,
                default_y_max=1,
                y_label_name="power (mW)",
            )
        ]
    )
    # -------------------------------------------------------------------

    config_source.write([visualizer_config])

def open_visualizer(width_proportion, height_proportion, receiver_endpoint):
    # start visualizer in a isolated process which is required on mac, instead of a sub process.
    gui_cmd = f"import samna, samnagui; samnagui.run_visualizer('{receiver_endpoint}', {width_proportion}, {height_proportion})"
    os_cmd = f'{sys.executable} -c "{gui_cmd}"'
    print("Visualizer start command: ", os_cmd)
    gui_thread = Thread(target=os.system, args=(os_cmd,))
    gui_thread.start()

    return gui_thread


streamer_endpoint = "tcp://0.0.0.0:40001"
width_proportion = 0.75
height_proportion = 0.75

gui_process = open_visualizer(width_proportion, height_proportion, streamer_endpoint)

dk = open_speck2f_dev_kit()

# route events
graph = samna.graph.EventFilterGraph()
build_samna_event_route(dk, graph, streamer_endpoint)

dvs_sink = samna.graph.sink_from(dk.get_model_source_node())

graph.start()

# modify configuration
config = samna.speck2f.configuration.SpeckConfiguration()
# enable dvs event monitoring
config.dvs_layer.monitor_enable = True
dk.get_model().apply_configuration(config)

# --- NEW: Start monitoring the power right before joining the process ---
# 100.0 is the sampling frequency in Hz
dk.get_power_monitor().start_auto_power_measurement(100.0)
# ------------------------------------------------------------------------

print("\n VISUALIZER IS LIVE ")
dvs_sink.get_events() # Flush old events
total_events = 0
start_time = time.time()

while gui_process.is_alive():
    events = dvs_sink.get_events()
    total_events += len(events)
    elapsed = time.time() - start_time
    print(f"\r  Time: {elapsed:.1f}s | Events Generated: {total_events:,}", end="", flush=True)
    time.sleep(0.05)
print()

# --- NEW: Stop the power measurement and graph on exit ---
dk.get_power_monitor().stop_auto_power_measurement()
graph.stop()
