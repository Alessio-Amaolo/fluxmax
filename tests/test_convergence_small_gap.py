"""Convergence in Fourier orders at fixed small gap.

Truncation and the d -> 0 limit work against each other: at small gap the
evanescent tail extends to k_par ~ 1/d, and the BZ-integral-plus-G-sum
convention only captures it once the retained G orders cover that range. The
FDT trace relation holds only when the retained mode set captures essentially
all of the emitter's absorption, so the truncation error at small gap is the
practical limit of the method.

This test pins the behavior: at gap = 0.1 lambda, the relative error of the
BZ-summed RCWA transfer against the k-integrated Polder-Van Hove reference must
fall by orders of magnitude as the truncation grows, and reach the tolerance of
the existing planar validation at the finest truncation.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fluxmax.physics.lifshitz as lifshitz
from fluxmax.physics import heat_transfer as ht
from fluxmax.setup import two_body as ss

jax.config.update("jax_enable_x64", True)


WAVELENGTH = 1.0
PITCH = 0.93  # do not make 1.0 because it falls on BZ boundary
GAP = 0.1
EPS_SLAB = 4.0 + 0.5j
THICKNESS = 0.5
BZ_GRID = (7, 7)  # fine enough that truncation, not BZ quadrature, is the limit
TERMS_SWEEP = [10, 50, 200]
FINAL_TOL = 5e-3
QUADRATURE_FLOOR = 2e-3
OMEGA = 2.0 * np.pi / WAVELENGTH


def _rcwa_transfer(num_terms: int) -> float:
    plv, expansion, ipw = ss.make_rcwa_setup(
        pitch=PITCH, approximate_num_terms=num_terms, brillouin_grid_shape=BZ_GRID
    )
    kw = dict(
        wavelength=jnp.asarray(WAVELENGTH),
        in_plane_wavevector=ipw,
        primitive_lattice_vectors=plv,
        expansion=expansion,
    )
    vac = ss.eigensolve_uniform(**kw, permittivity=1.0 + 0.0j)
    slab = ss.eigensolve_uniform(**kw, permittivity=EPS_SLAB)
    R_A, T_A, _ = ss.body_s_matrices(vac, slab, jnp.asarray(THICKNESS), is_body_A=True)
    R_B, T_B, _ = ss.body_s_matrices(vac, slab, jnp.asarray(THICKNESS), is_body_A=False)
    F_re, F_ah, F = ht.poynting_flux_matrices(vac)
    P = ht.propagation_matrix(vac.eigenvalues, GAP)
    sigma_A = ht.compute_sigma(R_A, T_A, F_re, F_ah)
    sigma_B = ht.compute_sigma(R_B, T_B, F_re, F_ah)
    tau = ht.spectral_transfer(sigma_A, sigma_B, P, R_A, R_B, F)
    n_bz = BZ_GRID[0] * BZ_GRID[1]
    area = float(np.asarray(ss.cell_area(plv)))
    return float(np.sum(np.real(np.asarray(tau))) / n_bz / area)


def test_convergence_in_orders_at_small_gap():
    reference = float(
        np.asarray(
            lifshitz.polder_van_hove_integrated(
                omega=OMEGA,
                eps_A=EPS_SLAB,
                thickness_A=THICKNESS,
                eps_B=EPS_SLAB,
                thickness_B=THICKNESS,
                gap=GAP,
                kpar_max_factor=50.0,
                n_kpar=8000,
            )
        )
    )
    errors = []
    for terms in TERMS_SWEEP:
        value = _rcwa_transfer(terms)
        errors.append(abs(value - reference) / abs(reference))
    # Each step must improve, unless it has already reached the quadrature floor
    # where more orders cannot help (see the module docstring).
    for (coarse_n, coarse), (fine_n, fine) in zip(
        zip(TERMS_SWEEP, errors), zip(TERMS_SWEEP[1:], errors[1:])
    ):
        assert fine < coarse or fine < QUADRATURE_FLOOR, (
            f"error grew from {coarse:.3e} at {coarse_n} terms to {fine:.3e} at "
            f"{fine_n} terms, and {fine:.3e} is above the {QUADRATURE_FLOOR:g} "
            f"quadrature floor, so this is a real loss of accuracy: {errors}"
        )
    assert errors[-1] < FINAL_TOL, (
        f"finest truncation ({TERMS_SWEEP[-1]} terms) error {errors[-1]:.3e} "
        f"exceeds {FINAL_TOL}; errors = {errors}"
    )
    # Teeth: the coarsest truncation must be far from the floor, otherwise the
    # sweep starts already converged and asserts nothing about truncation.
    assert errors[0] > 10 * QUADRATURE_FLOOR, (
        f"coarsest truncation ({TERMS_SWEEP[0]} terms) error {errors[0]:.3e} is "
        f"already near the {QUADRATURE_FLOOR:g} floor; the sweep has no teeth"
    )
