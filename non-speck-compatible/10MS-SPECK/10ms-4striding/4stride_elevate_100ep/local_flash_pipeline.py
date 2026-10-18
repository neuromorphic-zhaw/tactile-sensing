import os
import torch
import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork
from train_10ms_4striding_elevate import Speck9LayerTrainer  

# --- 1. LOCAL WORKSPACE SETUP ---
BATCH_SIZE = 1
MODEL_FILENAME = "4striding_elevate_100ep.pt"

if not os.path.exists(MODEL_FILENAME):
    raise FileNotFoundError(f"❌ Could not find model file '{MODEL_FILENAME}' in this folder.")

# --- 2. BUILD MODEL LAYOUT ARCHITECTURE ---
print("⚙️ Building model layout architecture...", flush=True)
model = Speck9LayerTrainer(target_batch_size=BATCH_SIZE)

dummy_data = torch.zeros((BATCH_SIZE, 40, 2, 128, 128))
with torch.no_grad():
    _ = model(dummy_data)

# --- 3. INGEST PARAMETERS ---
print(f"🔄 Mapping floating-point weights from {MODEL_FILENAME}...", flush=True)
model.load_state_dict(torch.load(MODEL_FILENAME, map_location="cpu"))
print("✅ Weights successfully read into system RAM.")

# --- 4. EXPORT TO CHIP SPECIFICATIONS ---
print("\n🗜️ Translating layers into 8-bit chip logic via DynapcnnNetwork...")
target_snn = model.spiking_backbone

hw_network = DynapcnnNetwork(
    snn=target_snn,
    discretize=True,
    input_shape=(2, 128, 128)  
)
print("✅ Parameter discretization complete.")

# --- 5. CONNECT AND FLASH VIA PHYSICAL DEVICE INFO MAPPING (WSL2 Robust) ---
print("\n⚡ Querying hardware devices on the virtual USB bus...")
devices = samna.device.get_unopened_devices()

if not devices:
    raise RuntimeError("❌ No connected Speck devices found. Run 'lsusb' to verify your usbipd attach bridge is active!")

# Capture the exact direct memory reference descriptor of your Speck2f board
target_device_info = None
for dev in devices:
    if "speck2f" in dev.model_name.lower():
        target_device_info = dev
        break

if target_device_info is None:
    raise RuntimeError(f"❌ Device found but it doesn't match Speck2f specifications. Found: {devices[0].model_name}")

print(f"✅ Secure Device Handle Acquired: {target_device_info.model_name} (ID: {target_device_info.device_id})")

try:
    # Pass the device structure info directly to avoid memory string parsing crashes
    hw_network.to(
        device=target_device_info,
        chip_layers_ordering="auto",
        monitor_layers=[-1]  
    )
    print("🎉 MODEL DEPLOYMENT SUCCESSFUL! The 9-core SNN is now live on physical silicon.")
    print(f"📍 Actual Hardware Allocation Mapping: {hw_network.chip_layers_ordering}")

except Exception as e:
    print(f"❌ Flash failure: {str(e)}")
    print("💡 Troubleshooting: Try resetting the usbipd connection from Windows PowerShell and re-run.")

# --- 6. WRITE FLASH FILE BINARY ---
print("\n💾 Compiling independent standalone boot image...")
samna_config = hw_network.get_configuration()
hardware_binary = samna.speck2f.configuration_to_flash_binary(samna_config)

with open("speck_standalone_tactile.bin", "wb") as f:
    f.write(bytes(hardware_binary))
print("📝 Independent flash image written to: 'speck_standalone_tactile.bin'")
