"""Test for the emitter-correlator transpose.

This is a check of the theory more than the numerics.

The FDT noise correlator as derived carries the transpose of the emitter's
absorption operator: tau = Tr[P+ D+ S_A D P (F+)^-1 S_B^T F^-1]. Sigma is Hermitian but
not real-symmetric, so the transpose matters exactly when the scattering mixes
modes (gratings); planar slabs have effectively diagonal Sigma and are blind
to it. This test checks:

  (i)   transposed and untransposed formulas differ on a mode-mixing structure,
  (ii)  only the transposed formula satisfies reciprocity tau_{A->B} = tau_{B->A},
  (iii) both agree (to machine precision) for planar slabs, where the existing
        Polder-Van Hove validation lives.

Reciprocity at fixed k_par requires the unit cell to have in-plane inversion
symmetry (verified separately; for e.g. laterally offset gratings only the
BZ-integrated flux is reciprocal), so the grating here uses centered stripes.
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from fluxmax.physics import heat_transfer as ht
from fluxmax.setup import two_body as ss

jax.config.update("jax_enable_x64", True)


WAVELENGTH = 1.0
PITCH = 1.0
GAP = 0.2
NUM_TERMS = 20
THICKNESS_A = 0.5
THICKNESS_B = 0.35
NX = 64
BZ_GRID = (3, 3)

EPS_A = 12.0 + 0.5j
EPS_B = 8.0 + 1.0j
EPS_PLANAR_A = 4.0 + 0.5j
EPS_PLANAR_B = 6.0 + 1.0j

RECIPROCITY_TOL = 1e-7
PLANAR_EQUALITY_TOL = 1e-12
MIN_GRATING_VIOLATION = 1e-3
MIN_TRANSPOSE_EFFECT = 1e-3


def _centered_grating(eps_stripe: complex, fill: float) -> jnp.ndarray:
    """1D grating along x (uniform in y) with a centered stripe, so the unit
    cell keeps the in-plane inversion symmetry required for fixed-k
    reciprocity."""
    x = (np.arange(NX) + 0.5) / NX
    offset = 0.5 - fill / 2
    profile = np.where(((x - offset) % 1.0) < fill, eps_stripe, 1.0 + 0.0j)
    return jnp.asarray(np.tile(profile[:, None], (1, NX)), dtype=complex)


def _build(spec_A, spec_B):
    """Assemble Sigma_A, Sigma_B, P, R_A, R_B, F for a two-body setup.

    Each spec is either a complex permittivity (uniform slab) or a 2D
    permittivity array (patterned slab).
    """
    plv, expansion, ipw = ss.make_rcwa_setup(
        pitch=PITCH,
        approximate_num_terms=NUM_TERMS,
        brillouin_grid_shape=BZ_GRID,
    )
    kw = dict(
        wavelength=jnp.asarray(WAVELENGTH),
        in_plane_wavevector=ipw,
        primitive_lattice_vectors=plv,
        expansion=expansion,
    )
    vac = ss.eigensolve_uniform(**kw, permittivity=1.0 + 0.0j)

    def _lsr(spec):
        if jnp.ndim(spec) == 0:
            return ss.eigensolve_uniform(**kw, permittivity=cast(complex, spec))
        return ss.eigensolve_patterned(**kw, permittivity_array=spec)

    R_A, T_A, _ = ss.body_s_matrices(
        vac, _lsr(spec_A), jnp.asarray(THICKNESS_A), is_body_A=True
    )
    R_B, T_B, _ = ss.body_s_matrices(
        vac, _lsr(spec_B), jnp.asarray(THICKNESS_B), is_body_A=False
    )
    F_re, F_ah, F = ht.poynting_flux_matrices(vac)
    P = ht.propagation_matrix(vac.eigenvalues, GAP)
    sigma_A = ht.compute_sigma(R_A, T_A, F_re, F_ah)
    sigma_B = ht.compute_sigma(R_B, T_B, F_re, F_ah)
    return sigma_A, sigma_B, P, R_A, R_B, F


def _transfer_untransposed(sigma_A, sigma_B, P, R_A, R_B, F):
    """The untransposed formula for comparison."""
    n = P.shape[-1]
    eye = jnp.eye(n, dtype=P.dtype)
    D = jnp.linalg.solve(eye - P @ R_B @ P @ R_A, eye)
    F_inv = jnp.linalg.solve(F, eye)
    corr = ht._adjoint(F_inv) @ sigma_B @ F_inv
    W = ht._adjoint(P) @ ht._adjoint(D) @ sigma_A @ D @ P @ corr
    return ht._trace(W)


def _max_reciprocity_violation(transfer_fn, sigma_A, sigma_B, P, R_A, R_B, F):
    """Max pointwise-in-k relative difference between A->B and B->A."""
    fwd = np.asarray(transfer_fn(sigma_A, sigma_B, P, R_A, R_B, F)).ravel()
    rev = np.asarray(transfer_fn(sigma_B, sigma_A, P, R_B, R_A, F)).ravel()
    scale = 0.5 * (np.abs(fwd) + np.abs(rev))
    return float(np.max(np.abs(fwd - rev) / np.where(scale > 0, scale, 1.0)))


def test_planar_reciprocity_and_transpose_insensitivity():
    """(iii) Planar slabs: reciprocal, and the transpose changes nothing."""
    pieces = _build(EPS_PLANAR_A, EPS_PLANAR_B)
    viol = _max_reciprocity_violation(ht.spectral_transfer, *pieces)
    assert viol < PLANAR_EQUALITY_TOL, f"planar reciprocity violated: {viol:.3e}"

    with_t = np.asarray(ht.spectral_transfer(*pieces)).ravel()
    without_t = np.asarray(_transfer_untransposed(*pieces)).ravel()
    rel = np.max(np.abs(with_t - without_t) / np.abs(with_t))
    assert rel < PLANAR_EQUALITY_TOL, f"planar transpose sensitivity: {rel:.3e}"


def test_grating_reciprocity_requires_transpose():
    """(i) + (ii) Mode-mixing structure: formulas differ, and only the
    transposed one is reciprocal."""
    pieces = _build(
        _centered_grating(EPS_A, fill=0.4), _centered_grating(EPS_B, fill=0.6)
    )

    viol_fixed = _max_reciprocity_violation(ht.spectral_transfer, *pieces)
    assert viol_fixed < RECIPROCITY_TOL, (
        f"transposed formula violates reciprocity on grating: {viol_fixed:.3e}"
    )

    viol_old = _max_reciprocity_violation(_transfer_untransposed, *pieces)
    assert viol_old > MIN_GRATING_VIOLATION, (
        "untransposed formula unexpectedly reciprocal on grating "
        f"({viol_old:.3e}); test structure may not mix modes"
    )

    with_t = np.asarray(ht.spectral_transfer(*pieces)).ravel()
    without_t = np.asarray(_transfer_untransposed(*pieces)).ravel()
    rel = np.max(np.abs(with_t - without_t) / np.abs(with_t))
    assert rel > MIN_TRANSPOSE_EFFECT, (
        f"transpose has no effect on grating ({rel:.3e}); "
        "test structure may not mix modes"
    )
