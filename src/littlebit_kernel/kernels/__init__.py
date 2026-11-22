"""Kernel variants. Each module exposes `forward(x, layer: DeviceLayer) -> mx.array`.

`KERNELS` maps a short name to that function; tests and benchmarks iterate
over it, so a new variant only needs to be registered here.
"""

from . import auto, fused, lut, lut2, lut3, lut4, lut5, naive, simd
from .common import DeviceLayer, DevicePath, round_layer_scales

KERNELS = {
    "naive": naive.forward,
    "simd": simd.forward,
    "lut": lut.forward,
    "lut2": lut2.forward,
    "lut3": lut3.forward,
    "lut4": lut4.forward,
    "lut5": lut5.forward,
    "auto": auto.forward,   # <- the one to use: lut5 for small layers, lut4 for big
    "fused": fused.forward,
}

__all__ = ["KERNELS", "DeviceLayer", "DevicePath", "round_layer_scales"]
