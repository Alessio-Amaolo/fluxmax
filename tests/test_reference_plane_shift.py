"""The trace formula must not care where the gap reference planes are put.

Padding body A with a vacuum spacer of thickness s on its gap side and
shrinking the gap to d - s describes the same physical system, so

    tau(A + vac(s), B, d - s) = tau(A, B, d)

must hold identically, and likewise for body B, and for both at once.
"""

import jax
import jax.numpy as jnp
import numpy as np
from fmmax.scattering import stack_s_matrix

from fluxmax.physics import heat_transfer as ht
from fluxmax.setup import two_body as ss

jax.config.update("jax_enable_x64", True)

WAVELENGTH = 1.0
PITCH = 1.0
GAP = 0.2
SHIFT_A = 0.07
SHIFT_B = 0.05
NUM_TERMS = 20
THICKNESS_A = 0.5
THICKNESS_B = 0.35
NX = 64
BZ_GRID = (3, 3)

EPS_GRATING_A = 12.0 + 0.5j
EPS_GRATING_B = 8.0 + 1.0j
EPS_PLANAR_A = 4.0 + 0.5j
EPS_PLANAR_B = 6.0 + 1.0j

INVARIANCE_TOL = 1e-9
SIGMA_TOL = 1e-9
MIN_SHIFT_EFFECT = 1e-3  # tau must actually move if the gap is not shrunk
MIN_UNTRANSPOSED_VIOLATION = 1e-4


def _grating(eps_stripe: complex, fill: float) -> jnp.ndarray:
    x = (np.arange(NX) + 0.5) / NX
    offset = 0.5 - fill / 2
    profile = np.where(((x - offset) % 1.0) < fill, eps_stripe, 1.0 + 0.0j)
    return jnp.asarray(np.tile(profile[:, None], (1, NX)), dtype=complex)


def _setup(spec_A, spec_B):
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

    def lsr(spec):
        if jnp.ndim(spec) == 0:
            return ss.eigensolve_uniform(**kw, permittivity=spec)
        return ss.eigensolve_patterned(**kw, permittivity_array=spec)

    return vac, lsr(spec_A), lsr(spec_B)


def _body_blocks(vac, slab, thickness, *, is_body_A, shift=0.0):
    """R and T of one body as seen from the gap, with the gap-side reference
    plane moved a distance ``shift`` into the gap.

    The reference plane is moved by inserting a vacuum layer of thickness
    ``shift`` between the body and the (zero-thickness) bounding vacuum layer
    that defines the plane, so both the forward and backward amplitudes are
    referenced to the shifted plane.
    """
    zero = jnp.zeros_like(jnp.asarray(thickness))
    spacer = jnp.asarray(shift, dtype=jnp.asarray(thickness).dtype)
    if is_body_A:  # gap on the right
        layers = [vac, slab, vac, vac]
        thicknesses = [zero, jnp.asarray(thickness), spacer, zero]
        s = stack_s_matrix(layer_solve_results=layers, layer_thicknesses=thicknesses)
        return s.s12, s.s22
    layers = [vac, vac, slab, vac]  # gap on the left
    thicknesses = [zero, spacer, jnp.asarray(thickness), zero]
    s = stack_s_matrix(layer_solve_results=layers, layer_thicknesses=thicknesses)
    return s.s21, s.s11


def _tau(vac, slab_A, slab_B, *, gap, shift_A=0.0, shift_B=0.0, transpose=True):
    R_A, T_A = _body_blocks(vac, slab_A, THICKNESS_A, is_body_A=True, shift=shift_A)
    R_B, T_B = _body_blocks(vac, slab_B, THICKNESS_B, is_body_A=False, shift=shift_B)
    F_re, F_ah, F = ht.poynting_flux_matrices(vac)
    P = ht.propagation_matrix(vac.eigenvalues, gap)
    sigma_A = ht.compute_sigma(R_A, T_A, F_re, F_ah)
    sigma_B = ht.compute_sigma(R_B, T_B, F_re, F_ah)
    if transpose:
        tau = ht.spectral_transfer(sigma_A, sigma_B, P, R_A, R_B, F)
    else:
        n = P.shape[-1]
        eye = jnp.eye(n, dtype=P.dtype)
        D = jnp.linalg.solve(eye - P @ R_B @ P @ R_A, eye)
        F_inv = jnp.linalg.solve(F, eye)
        corr = ht._adjoint(F_inv) @ sigma_B @ F_inv  # no transpose on Sigma_B
        tau = ht._trace(ht._adjoint(P) @ ht._adjoint(D) @ sigma_A @ D @ P @ corr)
    return np.real(np.asarray(tau)).ravel()


def _rel(a, b):
    scale = 0.5 * (np.abs(a) + np.abs(b))
    return float(np.max(np.abs(a - b) / np.where(scale > 0, scale, 1.0)))


def _cases():
    return {
        "planar": (EPS_PLANAR_A, EPS_PLANAR_B),
        "grating": (
            _grating(EPS_GRATING_A, fill=0.4),
            _grating(EPS_GRATING_B, fill=0.6),
        ),
    }


def test_tau_invariant_under_reference_plane_shift():
    for name, (spec_A, spec_B) in _cases().items():
        vac, slab_A, slab_B = _setup(spec_A, spec_B)
        base = _tau(vac, slab_A, slab_B, gap=GAP)

        shifted_A = _tau(vac, slab_A, slab_B, gap=GAP - SHIFT_A, shift_A=SHIFT_A)
        shifted_B = _tau(vac, slab_A, slab_B, gap=GAP - SHIFT_B, shift_B=SHIFT_B)
        shifted_both = _tau(
            vac,
            slab_A,
            slab_B,
            gap=GAP - SHIFT_A - SHIFT_B,
            shift_A=SHIFT_A,
            shift_B=SHIFT_B,
        )
        for tag, value in (
            ("A", shifted_A),
            ("B", shifted_B),
            ("both", shifted_both),
        ):
            rel = _rel(base, value)
            assert rel < INVARIANCE_TOL, (
                f"{name}: shifting {tag}'s reference plane changed tau by {rel:.3e}"
            )

        # Teeth: without shrinking the gap, tau must move appreciably.
        moved = _tau(vac, slab_A, slab_B, gap=GAP, shift_A=SHIFT_A)
        assert _rel(base, moved) > MIN_SHIFT_EFFECT, (
            f"{name}: padding without shrinking the gap left tau unchanged "
            "-- the invariance check above is vacuous"
        )


def test_sigma_transforms_as_congruence_under_plane_shift():
    """The microscopic reason for the invariance: Sigma is the absorbed-power
    operator, so moving its reference plane conjugates it with the gap
    propagator."""
    for name, (spec_A, spec_B) in _cases().items():
        vac, slab_A, slab_B = _setup(spec_A, spec_B)
        F_re, F_ah, _ = ht.poynting_flux_matrices(vac)
        P_shift_A = ht.propagation_matrix(vac.eigenvalues, SHIFT_A)
        P_shift_B = ht.propagation_matrix(vac.eigenvalues, SHIFT_B)

        for tag, slab, thickness, is_A, shift, P_shift in (
            ("A", slab_A, THICKNESS_A, True, SHIFT_A, P_shift_A),
            ("B", slab_B, THICKNESS_B, False, SHIFT_B, P_shift_B),
        ):
            R, T = _body_blocks(vac, slab, thickness, is_body_A=is_A)
            R_s, T_s = _body_blocks(vac, slab, thickness, is_body_A=is_A, shift=shift)
            sigma = ht.compute_sigma(R, T, F_re, F_ah)
            sigma_shifted = ht.compute_sigma(R_s, T_s, F_re, F_ah)
            expected = ht._adjoint(P_shift) @ sigma @ P_shift
            rel = float(
                jnp.max(jnp.abs(sigma_shifted - expected))
                / jnp.max(jnp.abs(sigma_shifted))
            )
            assert rel < SIGMA_TOL, (
                f"{name}, body {tag}: Sigma' != P^dag Sigma P ({rel:.3e})"
            )


def test_untransposed_correlator_breaks_plane_shift_invariance():
    """Plane-shift invariance singles out the transposed correlator.

    The transpose sits on the *emitter's* Sigma, which in
    :func:`~fluxmax.physics.heat_transfer.spectral_transfer` is the
    ``sigma_B`` argument -- so it is body B's reference plane that exposes it,
    and only for a body that mixes propagating with evanescent channels. With
    the untransposed correlator the emitter-side shift violates the identity by
    ~3e-2 on this grating, while body A's (absorber) shift stays invariant
    either way.

    Basically a check for the theory.
    """
    spec_A, spec_B = _cases()["grating"]
    vac, slab_A, slab_B = _setup(spec_A, spec_B)
    base = _tau(vac, slab_A, slab_B, gap=GAP, transpose=False)
    shifted_emitter = _tau(
        vac, slab_A, slab_B, gap=GAP - SHIFT_B, shift_B=SHIFT_B, transpose=False
    )
    rel = _rel(base, shifted_emitter)
    assert rel > MIN_UNTRANSPOSED_VIOLATION, (
        "the untransposed correlator was expected to break plane-shift "
        f"invariance for an emitter-side shift, but the violation is only "
        f"{rel:.3e}"
    )

    # The absorber-side shift is invariant for either correlator, so it cannot
    # discriminate; record that so the asymmetry above is not read as an
    # accident.
    shifted_absorber = _tau(
        vac, slab_A, slab_B, gap=GAP - SHIFT_A, shift_A=SHIFT_A, transpose=False
    )
    assert _rel(base, shifted_absorber) < INVARIANCE_TOL
