"""Parametric Speck-fitting backbone shared by all demo scripts.

Same forward contract as speck_compatible/devin/train_speck_fitting.py:
    input  (B, T, 2, 128, 128)  -> time-stride ::4 -> flatten to (B*T', C, H, W)
    output (B, n_classes)       -> mean-over-time spike-rate logits

`channels` holds the 8 conv-layer output-channel counts. Conv i reads
channels[i-1] inputs (conv 0 reads the 2 DVS polarity channels), so
growing one entry automatically rewires the next layer's in-channels.

MAX_CHANNELS is written in by maximize.py — the largest taper the real
core-mapping solver still accepts. Keep the real trainer at or below it.
"""

import torch
import torch.nn as nn
import sinabs
import sinabs.layers as sl

# proven taper (verified mapping {1:0,2:1,3:2,4:8,5:4,6:3,7:6,8:5,9:7})
BASE_CHANNELS = (16, 32, 32, 32, 32, 64, 64, 64)

# maximize.py result: largest taper the solver maps. Kernel memory is the
# wall — 6/9 cores sit at 100% kernel. Train at/below this, never above.
MAX_CHANNELS = (16, 64, 32, 32, 32, 64, 64, 64)


def _iaf(batch_size):
    return sl.IAFSqueeze(batch_size=batch_size, spike_threshold=1.0,
                         min_v_mem=-1.0,
                         surrogate_grad_fn=sinabs.activation.PeriodicExponential())


def build_backbone(channels=BASE_CHANNELS, n_classes=2, batch_size=1):
    """8 strided conv blocks + AvgPool + Linear readout == 9 DYNAP-CNN layers."""
    c = list(channels)
    convs = [
        (2,    c[0], 4, 2),  # 128 -> 64
        (c[0], c[1], 4, 2),  # 64  -> 32
        (c[1], c[2], 4, 2),  # 32  -> 16
        (c[2], c[3], 3, 1),  # 16  -> 16
        (c[3], c[4], 4, 2),  # 16  -> 8
        (c[4], c[5], 4, 2),  # 8   -> 4
        (c[5], c[6], 3, 1),  # 4   -> 4
        (c[6], c[7], 3, 1),  # 4   -> 4
    ]
    layers = []
    for cin, cout, k, s in convs:
        layers += [nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=1,
                             bias=False), _iaf(batch_size)]
    layers += [nn.AvgPool2d(2, 2),                      # 4x4 -> 2x2
               nn.Flatten(),
               nn.Linear(c[7] * 4, n_classes, bias=False),
               _iaf(batch_size)]
    backbone = nn.Sequential(*layers)
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    return backbone


class SpeckDemoTrainer(nn.Module):
    def __init__(self, channels=MAX_CHANNELS, n_classes=2, batch_size=1):
        super().__init__()
        self.channels = tuple(channels)
        self.spiking_backbone = build_backbone(channels, n_classes, batch_size)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x[:, ::4].reshape(b * (t // 4), c, h, w)
        return self.spiking_backbone(x).reshape(b, t // 4, -1).mean(dim=1)


def load_checkpoint(model, path):
    """Load a trainer state_dict, dropping transient v_mem buffers."""
    sd = torch.load(path, map_location="cpu")
    weights = {k: v for k, v in sd.items() if not k.endswith(".v_mem")}
    model.load_state_dict(weights, strict=False)
    return len(weights)
