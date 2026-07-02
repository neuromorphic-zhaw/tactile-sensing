import torch
import torch.nn as nn
import sinabs.layers as sl
# --- IMPORT THE ACTUAL PARSING COMPILER CONTAINER ---
from sinabs.backend.dynapcnn import DynapcnnNetwork

# --- 1. DEFINING YOUR PROPOSED 9-LAYER BLUEPRINT ---
test_pipeline = nn.Sequential(
    # Core 0: Downsample step (128x128 -> 64x64)
    nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False), 
    nn.BatchNorm2d(32), sl.IAFSqueeze(batch_size=1), 

    # Core 1: (64x64 -> 32x32)
    nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False), 
    nn.BatchNorm2d(48), sl.IAFSqueeze(batch_size=1), sl.SumPool2d(2),

    # Core 2: Bottleneck (32x32 -> 16x16)
    nn.Conv2d(48, 64, kernel_size=1, bias=False), 
    nn.BatchNorm2d(64), sl.IAFSqueeze(batch_size=1), sl.SumPool2d(2),

    # Core 3: (16x16)
    nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=False), 
    nn.BatchNorm2d(48), sl.IAFSqueeze(batch_size=1),

    # Core 4: (16x16 -> 8x8)
    nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=False), 
    nn.BatchNorm2d(64), sl.IAFSqueeze(batch_size=1), sl.SumPool2d(2),

    # Core 5: High-Memory Core (8x8)
    nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False), 
    nn.BatchNorm2d(96), sl.IAFSqueeze(batch_size=1),

    # Core 6: High-Memory Core (8x8 -> 4x4)
    nn.Conv2d(96, 128, kernel_size=2, padding=1, bias=False), 
    nn.BatchNorm2d(128), sl.IAFSqueeze(batch_size=1), sl.SumPool2d(2),

    # Core 7: Bottleneck (4x4)
    nn.Conv2d(128, 32, kernel_size=1, bias=False),
    nn.BatchNorm2d(32), sl.IAFSqueeze(batch_size=1),

    # Core 8: Readout (4x4 -> 1x1)
    nn.Conv2d(32, 2, kernel_size=4, bias=False),
    sl.IAFSqueeze(batch_size=1),
    nn.Flatten()
)

INPUT_SHAPE = (2, 128, 128)

# --- 2. EXECUTE SILICON CEILING CHECKER ---
print("🔍 Passing architecture into the Dynapcnn compiler engine...")
try:
    # Trigger the hardware placement router using random initialization parameters
    hw_model = DynapcnnNetwork(
        snn=test_pipeline,
        input_shape=INPUT_SHAPE,
        discretize=True # Enforces the strict 8-bit parameter boundaries
    )
    
    print("\n🟢 VERIFICATION SUCCESSFUL!")
    print("This topology satisfies all hardware kernel parameter and neuron memory boundaries.")
    print("You can safely proceed to train this model on the HPC cluster!")

except Exception as e:
    print("\n🔴 COMPATIBILITY CHECK FAILED!")
    print(f"Hardware Constraint Violation: {e}")