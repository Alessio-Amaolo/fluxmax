"""Sigma must be the absorption operator, including for mode-mixing bodies.

These two checks pin the sign of the evanescent cross term in
:func:`fluxmax.physics.heat_transfer.compute_sigma`, i.e. the
``-i(kappa R - R^dag kappa)`` term of Eq. (13) of the notes.

Planar geometries are blind to this sign: Sigma is diagonal, the sign appears
once in the absorber's Sigma and once in the emitter's, and the two cancel in
tau. So the whole Polder-Van Hove validation suite passes with either sign, and
only a body whose scattering *mixes* propagating and evanescent channels (a
grating) can tell them apart. Both tests below use a grating for that reason.
"""

import jax
import jax.numpy as jnp
import numpy as np
from fmmax.fields import directional_poynting_flux  # type: ignore[attr-defined]

from fluxmax.physics import heat_transfer as ht
from fluxmax.setup import two_body as ss

jax.config.update("jax_enable_x64", True)

WAVELENGTH = 1.0
PITCH = 1.0
GAP = 0.2
NX = 64
NUM_TERMS = 20
THICKNESS_A = 0.5
THICKNESS_B = 0.35
K_GENERIC = np.array([[0.37 * 2 * np.pi, 0.11 * 2 * np.pi]])

LOSSLESS_TOL = 1e-11  # relative to ||F||, resp. to the lossy tau
ABSORPTION_RTOL = 1e-9


def _grating(eps_stripe: complex, fill: float) -> jnp.ndarray:
    """Centered stripe grating, uniform along y."""
    x = (np.arange(NX) + 0.5) / NX
    offset = 0.5 - fill / 2
    profile = np.where(((x - offset) % 1.0) < fill, eps_stripe, 1.0 + 0.0j)
    return jnp.asarray(np.tile(profile[:, None], (1, NX)), dtype=complex)


def _build(eps_A, eps_B, k=K_GENERIC):
    plv, expansion, _ = ss.make_rcwa_setup(pitch=PITCH, approximate_num_terms=NUM_TERMS)
    kw = dict(
        wavelength=jnp.asarray(WAVELENGTH),
        in_plane_wavevector=jnp.asarray(k),
        primitive_lattice_vectors=plv,
        expansion=expansion,
    )
    vac = ss.eigensolve_uniform(**kw, permittivity=1.0 + 0.0j)
    lsr_A = ss.eigensolve_patterned(**kw, permittivity_array=eps_A)
    lsr_B = ss.eigensolve_patterned(**kw, permittivity_array=eps_B)
    R_A, T_A, _ = ss.body_s_matrices(
        vac, lsr_A, jnp.asarray(THICKNESS_A), is_body_A=True
    )
    R_B, T_B, _ = ss.body_s_matrices(
        vac, lsr_B, jnp.asarray(THICKNESS_B), is_body_A=False
    )
    F_re, F_ah, F = ht.poynting_flux_matrices(vac)
    return dict(
        vac=vac,
        R_A=R_A,
        T_A=T_A,
        R_B=R_B,
        T_B=T_B,
        F_re=F_re,
        F_ah=F_ah,
        F=F,
        sigma_A=ht.compute_sigma(R_A, T_A, F_re, F_ah),
        sigma_B=ht.compute_sigma(R_B, T_B, F_re, F_ah),
        P=ht.propagation_matrix(vac.eigenvalues, GAP),
    )


def test_lossless_grating_has_zero_sigma_and_zero_transfer():
    """Energy conservation: a real-permittivity body absorbs nothing, so Sigma
    vanishes identically and the transfer is zero -- even when the grating mixes
    propagating and evanescent orders."""
    m = _build(_grating(12.0 + 0.0j, 0.4), _grating(8.0 + 0.0j, 0.6))
    scale = float(jnp.linalg.norm(m["F_re"]) + jnp.linalg.norm(m["F_ah"]))

    for tag in ("sigma_A", "sigma_B"):
        rel = float(jnp.linalg.norm(m[tag])) / scale
        assert rel < LOSSLESS_TOL, f"lossless {tag} does not vanish: {rel:.3e}"

    tau = float(
        jnp.real(
            jnp.sum(
                ht.spectral_transfer(
                    m["sigma_A"], m["sigma_B"], m["P"], m["R_A"], m["R_B"], m["F"]
                )
            )
        )
    )
    lossy = _build(_grating(12.0 + 0.5j, 0.4), _grating(8.0 + 1.0j, 0.6))
    tau_lossy = float(
        jnp.real(
            jnp.sum(
                ht.spectral_transfer(
                    lossy["sigma_A"],
                    lossy["sigma_B"],
                    lossy["P"],
                    lossy["R_A"],
                    lossy["R_B"],
                    lossy["F"],
                )
            )
        )
    )
    assert abs(tau) < LOSSLESS_TOL * abs(tau_lossy), (
        f"lossless grating radiates: tau={tau:.6e} vs lossy tau={tau_lossy:.6e}"
    )


def test_sigma_is_the_absorbed_power_operator():
    """(1/2) b^dag Sigma b is the absorbed power, cross-checked against fmmax's
    own Poynting routine for an incident field with evanescent content."""
    m = _build(_grating(12.0 + 0.5j, 0.4), _grating(8.0 + 1.0j, 0.6))
    vac, R, T = m["vac"], m["R_B"], m["T_B"]
    n = np.asarray(R).shape[-1]

    rng = np.random.default_rng(0)
    b = jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))

    def net_flux(forward, backward):
        fwd, bwd = directional_poynting_flux(
            jnp.asarray(forward)[:, jnp.newaxis],
            jnp.asarray(backward)[:, jnp.newaxis],
            vac,
        )
        return float(jnp.real(jnp.sum(fwd) + jnp.sum(bwd)))

    # Body B is illuminated from the gap (its left face) by forward amplitude b:
    # absorbed = net flux entering the gap face - flux leaving the far face.
    R_sq = jnp.asarray(R)[0] if jnp.asarray(R).ndim == 3 else jnp.asarray(R)
    T_sq = jnp.asarray(T)[0] if jnp.asarray(T).ndim == 3 else jnp.asarray(T)
    p_abs = net_flux(b, R_sq @ b) - net_flux(T_sq @ b, jnp.zeros_like(b))

    sigma = jnp.asarray(m["sigma_B"])
    sigma = sigma[0] if sigma.ndim == 3 else sigma
    quad = float(jnp.real(jnp.conj(b) @ sigma @ b))

    assert p_abs > 0, f"reference absorbed power should be positive: {p_abs:.3e}"
    assert np.isclose(quad, 2.0 * p_abs, rtol=ABSORPTION_RTOL), (
        f"b^dag Sigma b = {quad:.10e} but 2 * P_abs = {2.0 * p_abs:.10e}"
    )
