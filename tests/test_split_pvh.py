"""N1: validate propagating and evanescent contributions SEPARATELY against the
two Polder-Van Hove expressions (review task N1; direct numerical test of the
|K|^-1 evanescent normalization derived in the notes' Appendix B, task C6).

The planar-sweep validation (test_validate_planar) only tests the k-integrated
sum. Here we compare the RCWA trace pointwise in k_par against the PVH
transmissivity, separately for |k_par| < w/c (propagating: the
(1-|R|^2-|T|^2)-type expression) and |k_par| > w/c (evanescent: the
4 Im(R_A) Im(R_B) e^{-2 kappa d} expression), then check the two region
integrals. The evanescent branch is the direct probe of the 1/(2|k_z|) = 1/(2 kappa)
normalization of the noise correlator: a wrong power of kappa (e.g. k_z = i kappa
in place of |k_z| without the compensating gauge) shows up here immediately.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import fluxmax.physics.lifshitz as lifshitz  # noqa: E402
from fluxmax.physics import heat_transfer as ht  # noqa: E402
from fluxmax.setup import two_body as ss  # noqa: E402

WAVELENGTH = 1.0
PITCH = 1.0
GAP = 0.2
EPS_A = 4.0 + 0.5j
EPS_B = 6.0 + 1.0j
THICKNESS_A = 0.5
THICKNESS_B = 0.35

N_KPAR = 40
KPAR_PROP_RANGE = (0.05, 0.95)   # in units of omega/c, away from k=0 and light line
KPAR_EVAN_RANGE = (1.05, 5.0)

POINTWISE_TOL = 1e-6
INTEGRATED_TOL = 1e-6

OMEGA = 2.0 * np.pi / WAVELENGTH  # c = 1


def _rcwa_tau(kpar_values: np.ndarray) -> np.ndarray:
    """RCWA trace at explicit k_par points with a single Fourier order, so each
    point carries exactly the physical G=0 channel (both polarizations)."""
    plv, expansion, _ = ss.make_rcwa_setup(pitch=PITCH, approximate_num_terms=1)
    ipw = jnp.stack(
        [jnp.asarray(kpar_values), jnp.zeros_like(jnp.asarray(kpar_values))], axis=-1
    )
    kw = dict(
        wavelength=jnp.asarray(WAVELENGTH),
        in_plane_wavevector=ipw,
        primitive_lattice_vectors=plv,
        expansion=expansion,
    )
    vac = ss.eigensolve_uniform(**kw, permittivity=1.0 + 0.0j)
    lsr_A = ss.eigensolve_uniform(**kw, permittivity=EPS_A)
    lsr_B = ss.eigensolve_uniform(**kw, permittivity=EPS_B)
    R_A, T_A, _ = ss.body_s_matrices(vac, lsr_A, jnp.asarray(THICKNESS_A), is_body_A=True)
    R_B, T_B, _ = ss.body_s_matrices(vac, lsr_B, jnp.asarray(THICKNESS_B), is_body_A=False)
    F_re, F_ah, F = ht.poynting_flux_matrices(vac)
    P = ht.propagation_matrix(vac.eigenvalues, GAP)
    sigma_A = ht.compute_sigma(R_A, T_A, F_re, F_ah)
    sigma_B = ht.compute_sigma(R_B, T_B, F_re, F_ah)
    tau = ht.spectral_transfer(sigma_A, sigma_B, P, R_A, R_B, F)
    return np.real(np.asarray(tau)).ravel()


def _pvh_tau(kpar_values: np.ndarray) -> np.ndarray:
    """Sum over both polarizations of the PVH per-mode transmissivity."""
    kz0 = lifshitz._kz(1.0 + 0j, OMEGA, jnp.asarray(kpar_values))
    total = np.zeros_like(kpar_values, dtype=float)
    for pol in ("s", "p"):
        R_A, T_A = lifshitz.slab_RT(EPS_A, OMEGA, jnp.asarray(kpar_values), THICKNESS_A, pol)
        R_B, T_B = lifshitz.slab_RT(EPS_B, OMEGA, jnp.asarray(kpar_values), THICKNESS_B, pol)
        total += np.asarray(
            lifshitz.polder_van_hove_per_mode(R_A, T_A, R_B, T_B, kz0, GAP)
        )
    return total


def _compare(region: str) -> tuple[float, float, float]:
    lo, hi = KPAR_PROP_RANGE if region == "prop" else KPAR_EVAN_RANGE
    kpar = np.linspace(lo, hi, N_KPAR) * OMEGA
    rcwa = _rcwa_tau(kpar)
    pvh = _pvh_tau(kpar)
    pointwise = float(np.max(np.abs(rcwa - pvh) / np.abs(pvh)))
    int_rcwa = float(np.trapezoid(kpar * rcwa, kpar))  # radial measure k dk
    int_pvh = float(np.trapezoid(kpar * pvh, kpar))
    integrated = abs(int_rcwa - int_pvh) / abs(int_pvh)
    return pointwise, integrated, int_pvh


def test_propagating_channels_match_pvh():
    pointwise, integrated, _ = _compare("prop")
    assert pointwise < POINTWISE_TOL, f"propagating pointwise err {pointwise:.3e}"
    assert integrated < INTEGRATED_TOL, f"propagating integrated err {integrated:.3e}"


def test_evanescent_channels_match_pvh():
    """The direct test of the evanescent-channel normalization (C6)."""
    pointwise, integrated, _ = _compare("evan")
    assert pointwise < POINTWISE_TOL, f"evanescent pointwise err {pointwise:.3e}"
    assert integrated < INTEGRATED_TOL, f"evanescent integrated err {integrated:.3e}"


if __name__ == "__main__":
    for region in ("prop", "evan"):
        pw, integ, tot = _compare(region)
        print(f"{region}: max pointwise rel err = {pw:.3e}, integrated rel err = {integ:.3e}, "
              f"PVH region integral = {tot:.6e}")
