import torch
import torch.nn as nn
import sinabs
import sinabs.layers as sl
from sinabs.layers import SumPool2d, IAFSqueeze
from sinabs.backend.dynapcnn import DynapcnnNetwork
import os

# --- 1. CONFIG ---
MODEL_PATH = "/cfs/earth/scratch/matheron/tactile-sensing/4_layer/clean100epoch/precision_polish/final_100ep_refine_acc86.67.pt"
INPUT_SHAPE = (2, 128, 128)
DEVICE = "cpu"

# --- 2. SILICON-COMPATIBLE ARCHITECTURE ---
class SpeckTactileSNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Calculates the spatial dimension reduction:
        # Input: 128x128
        # Core 1: Pool2d -> 64x64
        # Core 2: Pool2d -> 32x32
        # Core 3: Pool2d -> 16x16
        # Core 4: Pool2d -> 8x8
        
        self.model = nn.Sequential(
            # Core 1
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(16), 
            sl.IAFSqueeze(batch_size=1, spike_threshold=1.0), 
            sl.SumPool2d(kernel_size=2),

            # Core 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(32), 
            sl.IAFSqueeze(batch_size=1, spike_threshold=1.0), 
            sl.SumPool2d(kernel_size=2),

            # Core 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            sl.IAFSqueeze(batch_size=1, spike_threshold=1.0), 
            sl.SumPool2d(kernel_size=2),

            # Core 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            sl.IAFSqueeze(batch_size=1, spike_threshold=1.0), 
            sl.SumPool2d(kernel_size=2),

            # --- FIXED READOUT BLOCK (Core 5) ---
            # Replace nn.Flatten() and nn.Linear with a spatial-collapsing Convolution.
            # Using a kernel size of 4 perfectly reduces the remaining 4x4 space down to 1x1.
            # Output channels are set to 2 (representing Flat vs. Curved).
            nn.Conv2d(64, 2, kernel_size=4, bias=False),
            sl.IAFSqueeze(batch_size=1, spike_threshold=1.0),
            
            # Flatten is now completely safe at the final output boundary 
            # because the spatial parameters have already been processed in hardware.
            nn.Flatten() 
        )

    def forward(self, x):
        # Input x shape format: (Batch, Timesteps, Channels, Height, Width)
        batch_size, timesteps, C, H, W = x.shape
        
        # Reshape to squeeze Batch and Time together for IAF processing
        x = x.reshape(batch_size * timesteps, C, H, W)
        
        # Forward pass through the silicon-compatible layers
        output = self.model(x)  # Yields shape: (Batch * Timesteps, 2)
        
        # Reshape back to isolate time dimensions
        output = output.reshape(batch_size, timesteps, -1)
        
        # Accumulate the spiking outputs over the time window for final classification
        return output.sum(dim=1)


# --- 3. LOADING & CONVERSION ---
print(f"📦 Loading Model: {MODEL_PATH}")
model = SpeckTactileSNN().to(DEVICE)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Filter out membrane states
    filtered_dict = {k: v for k, v in checkpoint.items() if "v_mem" not in k}
    model.load_state_dict(filtered_dict, strict=False)
    model.eval()
    print("✅ Weights loaded.")
else:
    print(f"❌ ERROR: Model file {MODEL_PATH} not found!")
    exit(1)

print("⚙️ Converting to Speck (Quantizing to 8-bit)...")
try:
    # Build the hardware-mapped network
    dynapcnn_net = DynapcnnNetwork(
        snn=model.model, 
        input_shape=INPUT_SHAPE, 
        discretize=True
    )
    
    # --- 4. THE SAVE LOGIC ---
    OUTPUT_FILE = "tactile_speck_final.speck"
    
    import samna
    config = dynapcnn_net.make_config()
    samna.save(OUTPUT_FILE, config)
    print(f"🎉 SUCCESS! Model saved to {OUTPUT_FILE}")

except Exception as e:
    print(f"❌ CONVERSION FAILED: {e}")
    print("\n--- Running Memory Diagnostic Tool ---")
    try:
        from sinabs.backend.dynapcnn import utils
        # This will print a clear breakdown of kernel/neuron memory per layer
        utils.validate_memory_mapping_speck(model.model, input_shape=INPUT_SHAPE)
    except Exception as diag_err:
        print(f"Could not run diagnostic: {diag_err}")

print("✅ Finished.")