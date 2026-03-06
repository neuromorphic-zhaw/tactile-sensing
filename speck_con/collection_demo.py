import samna
import numpy as np
import h5py
import time
import sys
import os
from threading import Thread

def open_visualizer(width_proportion, height_proportion, receiver_endpoint):
    gui_cmd = f"import samna, samnagui; samnagui.run_visualizer('{receiver_endpoint}', {width_proportion}, {height_proportion})"
    os_cmd = f'{sys.executable} -c "{gui_cmd}"'
    gui_thread = Thread(target=os.system, args=(os_cmd,))
    gui_thread.start()
    return gui_thread

def record_gelsight_with_gui(duration=20, filename="trial_01.h5"):
    # 1. Start the GUI Thread First
    streamer_endpoint = "tcp://0.0.0.0:40001"
    gui_process = open_visualizer(0.75, 0.75, streamer_endpoint)

    # 2. Connect to the DevKit
    devices = samna.device.get_unopened_devices()
    if len(devices) == 0:
        raise Exception("No SynSense devices found!")
    
    board = samna.device.open_device(devices[0])
    print(f"\nConnected to: {devices[0].device_type_name}")

    # 3. Setup Recording Sinks
    dvs_sink = samna.graph.sink_from(board.get_model_source_node())
    power_sink = samna.graph.sink_from(board.get_power_monitor().get_source_node())

    # 4. Setup Visualizer Graph
    viz_graph = samna.graph.EventFilterGraph()
    _, _, streamer = viz_graph.sequential(
        [board.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"]
    )
    config_source, _ = viz_graph.sequential([samna.BasicSourceNode_ui_event(), streamer])

    streamer.set_streamer_endpoint(streamer_endpoint)
    while streamer.wait_for_receiver_count() == 0:
        time.sleep(0.1)

    viz_config = samna.ui.VisualizerConfiguration(
        plots=[samna.ui.ActivityPlotConfiguration(128, 128, "GelSight Real-Time DVS")]
    )
    config_source.write([viz_config])
    viz_graph.start()

    # 5. CONFIGURE THE CHIP
    config = samna.speck2f.configuration.SpeckConfiguration()
    config.dvs_layer.monitor_enable = True
    config.dvs_layer.raw_monitor_enable = True
    
    config.dvs_filter.enable = True
    config.dvs_filter.internal_slow_clk = True
    config.dvs_filter.hot_pixel_filter_enable = True
    config.dvs_filter.threshold = 0 
    
    board.get_model().apply_configuration(config)

    # =====================================================================
    # THE MANUAL TRIGGER
    # =====================================================================
    print("\n" + "="*50)
    print("👀 VISUALIZER IS LIVE 👀")
    print("Look at the window and test your GelSight.")
    print("="*50)
    
    # The script will pause here indefinitely until you press Enter
    input("\n>>> Press [ENTER] when you are ready to start recording... <<<")
    # =====================================================================

    # 6. Start Clocks and Power (Resets time to 0 right at the start)
    sw = board.get_stop_watch()
    sw.set_enable_value(True)
    sw.reset() 
    
    pm = board.get_power_monitor()
    pm.start_auto_power_measurement(20.0)

    # 7. The Recording Session
    print(f"\n🔴 RECORDING STARTED for {duration} seconds 🔴")
    dvs_sink.clear_events() # Clears out all the practice touches
    power_sink.clear_events()
    time.sleep(duration)
    
    # 8. Stop and Gather Data
    print("\n⏹️ RECORDING FINISHED. Processing data...")
    pm.stop_auto_power_measurement()
    raw_dvs_data = dvs_sink.get_events()
    raw_power_data = power_sink.get_events()

    # 9. Format for HDF5
    # Explicitly filter by exact event types to avoid catching neural Spikes
    dvs_events = [e for e in raw_dvs_data if isinstance(e, samna.speck2f.event.DvsEvent)]
    pwr_events = [e for e in raw_power_data if isinstance(e, samna.unifirm.modules.events.PowerMeasurement)]

    dvs_array = np.array(
        [(e.x, e.y, e.p, e.timestamp) for e in dvs_events],
        dtype=[('x', 'u1'), ('y', 'u1'), ('p', 'u1'), ('t', 'u8')]
    )

    pwr_array = np.array(
        [(e.value, e.channel, e.timestamp) for e in pwr_events],
        dtype=[('watt', 'f4'), ('chan', 'u1'), ('t', 'u8')]
    )

    with h5py.File(filename, "w") as f:
        f.create_dataset("events", data=dvs_array, compression="gzip")
        f.create_dataset("power", data=pwr_array, compression="gzip")
        f.attrs['setup'] = "Markerless GelSight with GUI"

    print(f"\n--- SUCCESS ---")
    print(f"File Saved: {filename}")
    print(f"DVS Events: {len(dvs_array)} | Power Samples: {len(pwr_array)}")
    
    # 10. Cleanup
    viz_graph.stop()
    print("\nYou can now close the Visualizer window to fully exit the script.")
    gui_process.join() 

if __name__ == "__main__":
    record_gelsight_with_gui(duration=20, filename="trial_01.h5")
