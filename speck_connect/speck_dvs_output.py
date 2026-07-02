import samna
import sys
import os
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

    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])

    streamer.set_streamer_endpoint(endpoint)
    if streamer.wait_for_receiver_count() == 0:
        raise Exception(f'connecting to visualizer on {endpoint} fails')

    visualizer_config = samna.ui.VisualizerConfiguration(
        plots=[samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer")]
    )

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

graph.start()

# modify configuration
config = samna.speck2f.configuration.SpeckConfiguration()
# enable dvs event monitoring
config.dvs_layer.monitor_enable = True
dk.get_model().apply_configuration(config)

# wait until visualizer window destroys.
gui_process.join()

graph.stop()
