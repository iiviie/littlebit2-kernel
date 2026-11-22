"""Kernel variants. Each module exposes `forward(x, layer: DeviceLayer) -> mx.array`.

`KERNELS` maps a short name to that function; tests and benchmarks iterate
over it, so a new variant only needs to be registered here.
"""

from . import naive
from .common import DeviceLayer, DevicePath, round_layer_scales

KERNELS = {
    "naive": naive.forward,
}

__all__ = ["KERNELS", "DeviceLayer", "DevicePath", "round_layer_scales"]
