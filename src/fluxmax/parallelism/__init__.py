"""Pluggable BZ-averaging execution strategies for RCWA k-point sums.

Provides three modes for distributing k-point work:

- ``single_device_direct``  - all k-points in one vectorised call.
- ``single_device_chunked`` - constant-memory ``lax.scan`` over chunks.
- ``multi_device_chunked``  - sharded ``lax.scan`` across JAX devices.

See :mod:`fluxmax.parallelism.execution` for the full API.
"""

from .execution import (
    VALID_MODES,
    KernelFn,
    compute_bz_average,
    flatten_k_points,
)

__all__ = [
    "KernelFn",
    "VALID_MODES",
    "flatten_k_points",
    "compute_bz_average",
]
