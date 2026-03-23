# tactile-sensing
An ultra-low-power, event-based tactile sensor integrating a markerless GelSight with the SynSense Speck neuromorphic chip for high-speed force and slip detection.

The /speck_con folder contains all output visualizer scripts.
dvs_power_out.py displays both the events + power outputs of the speck.

collect_gzip_curv.py is the main curvature data collection script which includes a GUI with Events and Power as well as a terminal event generation moniter to kepp an eye on the amount of total events generated. The script also compresses the .h5 file with GZIP, for easier data handling.
