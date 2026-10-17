# Project: Tactile Sensing with Synsense Speck

## Overview
Ultra-low-power, event-based tactile sensor integrating a markerless GelSight with the SynSense Speck neuromorphic chip for high-speed force, texture and curvature detection.

## Environments

### Primary: WSL (Ubuntu)
- **Path**: /home/ronald/tactile-sensing
- **Python Environment**: /home/ronald/speck (virtualenv)
- **Hardware**: Synsense Speck connected via USB passthrough
- **CUDA**: Available (v12.8)

### Secondary: University HPC
- **Host**: login-rhel8.hpc.zhaw.ch
- **User**: matheron
- **Purpose**: Model training, dataset storage, heavy computation
- **Remote Path**: ~/tactile-sensing (if synced)

### Tertiary: Windows
- **WSL Access**: Via wsl bash -c "cd /home/ronald/tactile-sensing"
- **Purpose**: Development, testing, hardware debugging

## Key Dependencies
- PyTorch (CUDA 12.8)
- samna (Speck SDK)
- sinabs (Spiking neural networks)
- tonic (Neuromorphic datasets)
- h5py (Data storage)
- librosa (Audio processing)

## Project Structure
- speck_compatible/: Speck-compatible code
- speck_con/: Output visualizer scripts (dvs_power_out.py)
- speck_connect/: Connection utilities
- curv_data_collection/: Curvature data collection (collect_gzip_curv.py)
- realtime_live/: Real-time processing
- train_curv.py: Standard training script
- train_curv_ngc.py: NGC training script

## Common Commands

### Activate Environment
```bash
cd /home/ronald/tactile-sensing
source /home/ronald/speck/bin/activate
```

### Test Setup
```bash
python -c "import torch; import samna; import sinabs; import tonic; print('OK')"
```

### Run Training
```bash
python train_curv.py
# or
python train_curv_ngc.py
```

### Data Collection
```bash
python curv_data_collection/collect_gzip_curv.py
```

### Visualizer
```bash
python speck_con/dvs_power_out.py
```

### Connect to University HPC
```bash
ssh matheron@login-rhel8.hpc.zhaw.ch
```

## Hardware Setup (WSL)
1. Enable USB device passthrough in WSL
2. Connect Speck device
3. Verify: wsl bash -c "lsusb"
4. Set permissions: sudo chmod 666 /dev/bus/usb/*/*
5. Test with samna scripts

## Notes
- Data files are compressed with GZIP
- The project uses event-based neuromorphic computing
- CUDA is available for GPU acceleration
- Virtualenv is pre-configured at /home/ronald/speck
