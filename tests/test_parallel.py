"""Tests that all three BZ-averaging execution modes produce identical results.

A circular inclusion structure is used because planar structures converge
too quickly.

The multi-device test forces JAX to expose 4 virtual CPU devices so that
the sharded code path is exercised even on a single physical CPU.
"""

from __future__ import annotations

import os

# ── Force multiple virtual CPU devices BEFORE importing JAX ──────
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from fluxmax.optimization.design_tools import (
    circular_inclusion_permittivity,
)
from fluxmax.parallelism import compute_bz_average, flatten_k_points
from fluxmax.physics.kernels import (
    broadcast_slab_permittivity,
    make_two_body_bz_kernel,
)
from fluxmax.setup.two_body import make_rcwa_setup

PITCH = 1.0
DIAMETER = 0.5
EPS_HOST = 1.0 + 0.0j
EPS_INCLUSION = 4.0 + 0.5j
SLAB_THICKNESS = 0.5
GAP = 0.5
EPS_GAP = 1.0 + 0.0j
NUM_TERMS = 10
BZ_GRID = (3, 3)
RESOLUTION = PITCH / 16
OMEGAS = jnp.array([0.1, 0.2, 0.3])


@pytest.fixture(scope="module")
def rcwa_setup():
    plv, expansion, in_plane_wavevector = make_rcwa_setup(
        pitch=PITCH,
        approximate_num_terms=NUM_TERMS,
        brillouin_grid_shape=BZ_GRID,
    )
    return plv, expansion, in_plane_wavevector


@pytest.fixture(scope="module")
def eps_pattern():
    return circular_inclusion_permittivity(
        pitch=PITCH,
        diameter=DIAMETER,
        eps_host=EPS_HOST,
        eps_inclusion=EPS_INCLUSION,
        resolution=RESOLUTION,
    )


@pytest.fixture(scope="module")
def kernel_and_kpts(rcwa_setup, eps_pattern):
    plv, expansion, in_plane_wavevector = rcwa_setup
    kernel = make_two_body_bz_kernel(
        primitive_lattice_vectors=plv,
        expansion=expansion,
        slab_thickness=SLAB_THICKNESS,
        gap=GAP,
        eps_gap=EPS_GAP,
    )
    k_points = flatten_k_points(in_plane_wavevector)
    eps_omega = broadcast_slab_permittivity(eps_pattern, len(OMEGAS))
    return kernel, k_points, eps_omega


@pytest.fixture(scope="module")
def reference_direct(kernel_and_kpts):
    kernel, k_points, eps_omega = kernel_and_kpts
    return compute_bz_average(
        kernel,
        OMEGAS,
        eps_omega,
        k_points,
        execution_mode="single_device_direct",
    )


def test_direct_vs_chunked(kernel_and_kpts, reference_direct):
    """single_device_chunked must match single_device_direct."""
    kernel, k_points, eps_omega = kernel_and_kpts
    # n_k = 9 (3x3 BZ), k_chunk_size = 3 divides evenly
    result = compute_bz_average(
        kernel,
        OMEGAS,
        eps_omega,
        k_points,
        execution_mode="single_device_chunked",
        k_chunk_size=3,
    )
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference_direct), rtol=1e-10)


def test_direct_vs_sharded(kernel_and_kpts, reference_direct):
    """multi_device_chunked (4 virtual CPUs) must match single_device_direct."""
    kernel, k_points, eps_omega = kernel_and_kpts
    # 4 devices, global_chunk_size=4 divides evenly into devices.
    result = compute_bz_average(
        kernel,
        OMEGAS,
        eps_omega,
        k_points,
        execution_mode="multi_device_chunked",
        k_chunk_size=4,
    )
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference_direct), rtol=1e-10)


def test_omega_chunk_matches_direct(kernel_and_kpts, reference_direct):
    """omega_chunk_size=1 must match all-at-once vmap."""
    kernel, k_points, eps_omega = kernel_and_kpts
    result = compute_bz_average(
        kernel,
        OMEGAS,
        eps_omega,
        k_points,
        execution_mode="single_device_direct",
        omega_chunk_size=1,
    )
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference_direct), rtol=1e-10)


def test_omega_and_k_chunk_matches_direct(kernel_and_kpts, reference_direct):
    """Chunking both omega (size=1) and k (size=3) must match direct."""
    kernel, k_points, eps_omega = kernel_and_kpts
    result = compute_bz_average(
        kernel,
        OMEGAS,
        eps_omega,
        k_points,
        execution_mode="single_device_chunked",
        k_chunk_size=3,
        omega_chunk_size=1,
    )
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference_direct), rtol=1e-10)


def test_sharded_omega_and_k_chunk_matches_direct(kernel_and_kpts, reference_direct):
    """Chunking both omega (size=1) and k (size=4) must match direct."""
    kernel, k_points, eps_omega = kernel_and_kpts
    result = compute_bz_average(
        kernel,
        OMEGAS,
        eps_omega,
        k_points,
        execution_mode="multi_device_chunked",
        k_chunk_size=4,
        omega_chunk_size=1,
    )
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference_direct), rtol=1e-10)


def test_all_modes_nonzero(kernel_and_kpts):
    """Sanity check: results are finite and contain positive values."""
    kernel, k_points, eps_omega = kernel_and_kpts
    for mode, kw in [
        ("single_device_direct", {}),
        ("single_device_chunked", {"k_chunk_size": 3}),
        ("multi_device_chunked", {"k_chunk_size": 4}),
    ]:
        result = compute_bz_average(
            kernel,
            OMEGAS,
            eps_omega,
            k_points,
            execution_mode=mode,
            **kw,
        )
        assert jnp.all(jnp.isfinite(result)), f"{mode}: non-finite values"
        assert jnp.any(result > 0), f"{mode}: no positive values"


def test_chunked_rejects_bad_chunk_size(kernel_and_kpts):
    """k_chunk_size must divide n_k for chunked mode."""
    kernel, k_points, eps_omega = kernel_and_kpts
    with pytest.raises(ValueError, match="must be a multiple"):
        compute_bz_average(
            kernel,
            OMEGAS,
            eps_omega,
            k_points,
            execution_mode="single_device_chunked",
            k_chunk_size=7,  # does not divide 9
        )
