"""Brillouin-zone averaging with pluggable execution strategies.

The three k-point strategies distribute k-point work differently:

- direct: evaluate all k-points in one vectorised call.
- chunked: iterate over k-point and omega-point chunks with ``jax.lax.scan``
  (constant-memory on one device).
- sharded: pad, mask, and shard k-point and omega-point chunks across multiple
  JAX devices via ``Mesh`` / ``NamedSharding``.

This gives two independent knobs:

- ``omega_chunk_size``: how many omegas to vmap in parallel
  (``None`` = all omegas at once).
- ``k_chunk_size``: how many k-points per scan / shard iteration.

The kernel function has the signature::

    kernel_fn(omega_i, eps_i, k_points)
        -> Float[Array, "n_k"]

One scalar omega, one 2-D permittivity slice, one k-point batch.
The parallelism module handles the omega and k loops around it.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Kernel type: (omega_scalar, eps_2d, k_points) -> (n_k,)
KernelFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]

VALID_MODES = frozenset({
    "single_device_direct",
    "single_device_chunked",
    "multi_device_chunked",
})


def flatten_k_points(in_plane_wavevectors: jnp.ndarray) -> jnp.ndarray:
    """Reshape an arbitrary k-point grid to ``(n_k, 2)``."""
    return jnp.reshape(jnp.asarray(in_plane_wavevectors), (-1, 2))


# Internal batched-kernel type
# Built by the dispatcher: vmap(kernel_fn, omega_chunk) → fn(k_pts) -> (chunk, n_k)
_BatchedKernelFn = Callable[[jnp.ndarray], jnp.ndarray]


def _make_batched_kernel(
    kernel_fn: KernelFn,
    omega_chunk: jnp.ndarray,
    eps_chunk: jnp.ndarray,
) -> _BatchedKernelFn:
    """Wrap a single-omega kernel to vmap over an omega chunk."""
    def batched(k_points: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda w, e: kernel_fn(w, e, k_points))(
            omega_chunk, eps_chunk,
        )
    return batched


# Direct
def _bz_average_direct(
    batched_kernel: _BatchedKernelFn,
    k_points: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate kernel on all k-points at once and average."""
    vals = batched_kernel(k_points)  # (*batch, n_k)
    return jnp.mean(vals, axis=-1)


# Single-device chunked
def _bz_average_chunked(
    batched_kernel: _BatchedKernelFn,
    k_points: jnp.ndarray,
    k_chunk_size: int,
) -> jnp.ndarray:
    """Average over k-points using ``jax.lax.scan`` in fixed-size chunks."""
    n_k = int(k_points.shape[0])

    if k_chunk_size < 1:
        raise ValueError(f"k_chunk_size must be >= 1, got {k_chunk_size}")
    if n_k % k_chunk_size != 0:
        raise ValueError(
            f"n_k ({n_k}) must be a multiple of k_chunk_size ({k_chunk_size})."
        )

    num_chunks = n_k // k_chunk_size
    k_chunks = k_points.reshape(num_chunks, k_chunk_size, 2)

    @jax.checkpoint
    def process_chunk(k_chunk):
        tau_chunk = batched_kernel(k_chunk)
        return jnp.sum(tau_chunk, axis=-1)

    # Each chunk is independent, so use map instead of scan.
    # With @jax.checkpoint, the backward pass recomputes one chunk
    # at a time, giving constant peak memory in the number of chunks.
    chunk_sums = jax.lax.map(process_chunk, k_chunks)
    return jnp.sum(chunk_sums, axis=0) / n_k


# Multi-device sharded
def _bz_average_sharded(
    batched_kernel: _BatchedKernelFn,
    k_points: jnp.ndarray,
    global_chunk_size: int,
) -> jnp.ndarray:
    """Average over k-points sharded across all available JAX devices."""
    n_k_real = int(k_points.shape[0])

    if global_chunk_size < 1:
        raise ValueError(
            f"global_chunk_size must be >= 1, got {global_chunk_size}"
        )

    devices = jax.devices()
    n_devices = len(devices)
    if global_chunk_size % n_devices != 0:
        raise ValueError(
            "global_chunk_size must be divisible by the number of devices. "
            f"Got global_chunk_size={global_chunk_size}, n_devices={n_devices}."
        )

    # Pad k-points
    remainder = n_k_real % global_chunk_size
    pad_amount = (global_chunk_size - remainder) if remainder != 0 else 0

    if pad_amount > 0:
        padding = jnp.zeros((pad_amount, 2), dtype=k_points.dtype)
        k_points_padded = jnp.concatenate([k_points, padding], axis=0)
    else:
        k_points_padded = k_points

    n_k_padded = int(k_points_padded.shape[0])
    num_chunks = n_k_padded // global_chunk_size
    k_chunks = k_points_padded.reshape(num_chunks, global_chunk_size, 2)

    mask = jnp.arange(n_k_padded) < n_k_real
    mask_chunks = mask.reshape(num_chunks, global_chunk_size)

    # Per-chunk function
    @jax.checkpoint
    def process_chunk(inputs):
        k_chunk, mask_chunk = inputs
        tau_chunk = batched_kernel(k_chunk)
        tau_masked = tau_chunk * mask_chunk
        return jnp.sum(tau_masked, axis=-1)

    # Mesh + shardings
    mesh = Mesh(np.array(devices), axis_names=("data",))
    k_sharding = NamedSharding(mesh, P(None, "data", None))
    mask_sharding = NamedSharding(mesh, P(None, "data"))

    jitted_map = jax.jit(
        lambda k_in, mask_in: jax.lax.map(
            process_chunk, (k_in, mask_in)
        ),
        in_shardings=(k_sharding, mask_sharding),
    )

    chunk_sums = jitted_map(k_chunks, mask_chunks)
    return jnp.sum(chunk_sums, axis=0) / n_k_real


# Internal dispatcher for a single omega chunk
def _dispatch_k_strategy(
    batched_kernel: _BatchedKernelFn,
    k_points: jnp.ndarray,
    execution_mode: str,
    k_chunk_size: int,
) -> jnp.ndarray:
    if execution_mode == "single_device_direct":
        return _bz_average_direct(batched_kernel, k_points)
    if execution_mode == "single_device_chunked":
        return _bz_average_chunked(batched_kernel, k_points, k_chunk_size)
    if execution_mode == "multi_device_chunked":
        return _bz_average_sharded(batched_kernel, k_points, k_chunk_size)
    raise ValueError(
        f"Unsupported execution_mode {execution_mode!r}. "
        f"Supported modes: {sorted(VALID_MODES)}."
    )


# Public dispatcher
def compute_bz_average(
    kernel_fn: KernelFn,
    omega_1d: jnp.ndarray,
    eps_omega: jnp.ndarray,
    k_points: jnp.ndarray,
    *,
    execution_mode: str = "single_device_direct",
    k_chunk_size: int = 1,
    omega_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Compute BZ-averaged transfer for each frequency.

    Parameters
    ----------
    kernel_fn : KernelFn
        Geometry kernel:
        ``kernel(omega_i, eps_i, k_points) -> (n_k,)`` reals.
    omega_1d : jnp.ndarray, shape ``(n_omega,)``
        Angular frequencies.
    eps_omega : jnp.ndarray, shape ``(n_omega, ny, nx)``
        Per-frequency permittivity slices.
    k_points : jnp.ndarray
        In-plane wavevectors (flattened to ``(n_k, 2)``).
    execution_mode : str
        One of ``"single_device_direct"``, ``"single_device_chunked"``,
        or ``"multi_device_chunked"``.
    k_chunk_size : int
        Chunk size for chunked / sharded k-point modes.
    omega_chunk_size : int or None
        How many frequencies to vmap in parallel.  ``None`` (default)
        vmaps **all** omegas at once (equivalent to setting this equal
        to ``n_omega``).  Set to ``1`` to process one frequency at a
        time (minimum memory).

        For ``single_device_direct`` and ``single_device_chunked``
        modes the omega loop uses ``jax.lax.map`` (JIT-compilable
        sequential map).  For ``multi_device_chunked`` a Python loop
        is used because the sharded scan contains an internal
        ``jax.jit``.

    Returns
    -------
    jnp.ndarray, shape ``(n_omega,)``
    """
    k_points = flatten_k_points(k_points)
    omega_1d = jnp.asarray(omega_1d)
    eps_omega = jnp.asarray(eps_omega, dtype=complex)

    n_omega = int(omega_1d.shape[0])

    if omega_chunk_size is None:
        omega_chunk_size = n_omega  # (vmap everything at once)

    if omega_chunk_size < 1:
        raise ValueError(
            f"omega_chunk_size must be >= 1, got {omega_chunk_size}"
        )
    if n_omega % omega_chunk_size != 0:
        raise ValueError(
            f"n_omega ({n_omega}) must be a multiple of "
            f"omega_chunk_size ({omega_chunk_size})."
        )

    # Device-count sanity check for sharded mode
    if execution_mode == "multi_device_chunked":
        n_devices = len(jax.devices())
        n_k = int(k_points.shape[0])
        n_omega_chunks = n_omega // omega_chunk_size
        # Each sharded scan iteration distributes k_chunk_size across
        # devices, so the total parallel work per omega chunk is
        # ceil(n_k / k_chunk_size) iterations.  Warn when the product
        # of omega- and k-chunks doesn't utilise devices evenly.
        n_k_chunks = int(np.ceil(n_k / k_chunk_size))
        total_chunks = n_omega_chunks * n_k_chunks
        if total_chunks % n_devices != 0:
            import warnings
            warnings.warn(
                f"Total work chunks ({n_omega_chunks} omega × "
                f"{n_k_chunks} k = {total_chunks}) is not divisible "
                f"by the number of devices ({n_devices}).  Consider "
                f"adjusting omega_chunk_size or k_chunk_size for even "
                f"device utilisation.",
                stacklevel=2,
            )

    # Reshape omega/eps into (n_chunks, chunk_size, ...)
    num_omega_chunks = n_omega // omega_chunk_size
    omega_chunks = omega_1d.reshape(num_omega_chunks, omega_chunk_size)
    eps_chunks = eps_omega.reshape(
        num_omega_chunks, omega_chunk_size, *eps_omega.shape[1:],
    )

    @jax.checkpoint
    def _process_one_chunk(pair):
        omega_c, eps_c = pair
        batched = _make_batched_kernel(kernel_fn, omega_c, eps_c)
        return _dispatch_k_strategy(
            batched, k_points, execution_mode, k_chunk_size,
        )

    if execution_mode == "multi_device_chunked":
        # Sharded path calls jax.jit internally, which cannot be
        # nested inside lax.map.  Use a Python loop instead.
        results = []
        for i in range(num_omega_chunks):
            results.append(
                _process_one_chunk((omega_chunks[i], eps_chunks[i]))
            )
        return jnp.concatenate(results, axis=0)

    # For direct / chunked the inner functions are pure JAX, so
    # lax.map gives us a JIT-compilable sequential map.
    return jax.lax.map(_process_one_chunk, (omega_chunks, eps_chunks)).reshape(-1)

# Note! If we also want to average of integrate over omega, we could 
# do this more efficiently instead of having a python for loop.
# Something to keep in find for the future.

__all__ = [
    "KernelFn",
    "VALID_MODES",
    "flatten_k_points",
    "compute_bz_average",
]
