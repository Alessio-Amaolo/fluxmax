"""RCWA trace vs the analytic Polder-Van Hove reference, for planar bodies.

Planar bodies have a closed form (`fluxmax.physics.lifshitz`), so the trace can be
checked. Seven tests, each covering a different thing:

  propagating / evanescent split   pointwise + region integral, 1 order
  spectral sweep                   pointwise over 200 k_par, 3 gaps, + plot
  generic azimuth                  off the k_x axis, where F is not diagonal
  multi-order channel sum          9/21/49 orders, tau = sum_G tau_PVH(|k+G|)
  z-asymmetric bilayer             pins the S-matrix block convention
  truncation convergence           BZ-summed, small gap

Why each is needed:

* The evanescent branch is the direct probe of the 1/(2|k_z|) normalization of the
  noise correlator.
* At k_par = (k, 0) fmmax's (Hx, Hy) channels coincide with TE/TM, so every 2x2
  block of F, Sigma and the correlator is diagonal.
* One order gives two channels. With N orders a uniform slab keeps the G channels
  decoupled, so the trace must equal the channel-by-channel PVH sum
* Every other body in the suite is a single layer, hence mirror-symmetric in z.
* Truncation and d -> 0 fight each other: at small gap the evanescent tail reaches
  k_par ~ 1/d, and the trace relation holds only once the retained G orders cover
  it.
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _helpers import stack_blocks

import fluxmax.physics.lifshitz as lifshitz
from fluxmax.physics import heat_transfer as ht
from fluxmax.setup import two_body as ss

WAVELENGTH = 1.0
OMEGA = 2.0 * np.pi / WAVELENGTH
PITCH = 1.0  # safe: these tests drive explicit k, nearest channel |k+G|/omega ~ 0.96
PITCH_BZ = 0.93  # do not make 1.0 because it falls on BZ boundary
GAP = 0.2

EPS_A = 4.0 + 0.5j
EPS_B = 6.0 + 1.0j
THICKNESS_A = 0.5
THICKNESS_B = 0.35

# Bilayer pair: body X = (far sublayer, near sublayer), "near" faces the gap.
EPS_A_NEAR, THICK_A_NEAR = 4.0 + 0.5j, 0.30
EPS_A_FAR, THICK_A_FAR = 9.0 + 2.0j, 0.20
EPS_B_NEAR, THICK_B_NEAR = 6.0 + 1.0j, 0.25
EPS_B_FAR, THICK_B_FAR = 2.5 + 0.2j, 0.35

SPLIT_TOL = 1e-6  # pointwise and region-integral, prop/evan split
SPECTRAL_TOL = 1e-6
AZIMUTH_TOL = 1e-8
BILAYER_TOL = 1e-9
MULTI_ORDER_TOL = 1e-14
LIGHT_LINE_MARGIN = 5e-3
MIN_ASYMMETRY = 1e-2

N_KPAR_SPLIT = 40
KPAR_PROP_RANGE = (0.05, 0.95)  # units of omega/c, away from k=0 and the light line
KPAR_EVAN_RANGE = (1.05, 5.0)

SPECTRAL_GAPS = [0.05, 0.2, 1.0]
N_KPAR_SPECTRAL = 200
KPAR_MIN, KPAR_MAX = 0.02, 10.0
LIGHTLINE_EXCLUSION = 0.02  # skip +-2% around |k| = omega/c

CONVERGENCE_GAP = 0.1
EPS_CONVERGENCE = 4.0 + 0.5j  # same slab both sides
THICKNESS_CONVERGENCE = 0.5
BZ_GRID = (7, 7)  # fine enough that truncation, not BZ quadrature, is the limit
TERMS_SWEEP = [10, 50, 200]
CONVERGENCE_FINAL_TOL = 5e-3
QUADRATURE_FLOOR = 1e-3

OUTPUT_DIR = Path(__file__).resolve().parent / "test_output"
PLOT_PATH = OUTPUT_DIR / "spectral_kpar_comparison.png"


def _on_axis(kpar) -> np.ndarray:
    """k_par values on the k_x axis, as (n, 2) in-plane wavevectors."""
    kpar = np.asarray(kpar)
    return np.stack([kpar, np.zeros_like(kpar)], axis=-1)


def _rcwa_tau(k_points, *, num_terms: int = 1, gap: float = GAP):
    """tau at explicit in-plane wavevectors. Returns (tau, expansion)."""
    plv, expansion, _ = ss.make_rcwa_setup(pitch=PITCH, approximate_num_terms=num_terms)
    kw = dict(
        wavelength=jnp.asarray(WAVELENGTH),
        in_plane_wavevector=jnp.asarray(k_points),
        primitive_lattice_vectors=plv,
        expansion=expansion,
    )
    vac = ss.eigensolve_uniform(**kw, permittivity=1.0 + 0.0j)
    lsr_A = ss.eigensolve_uniform(**kw, permittivity=EPS_A)
    lsr_B = ss.eigensolve_uniform(**kw, permittivity=EPS_B)
    R_A, T_A, _ = ss.body_s_matrices(
        vac, lsr_A, jnp.asarray(THICKNESS_A), is_body_A=True
    )
    R_B, T_B, _ = ss.body_s_matrices(
        vac, lsr_B, jnp.asarray(THICKNESS_B), is_body_A=False
    )
    F_re, F_ah, F = ht.poynting_flux_matrices(vac)
    P = ht.propagation_matrix(vac.eigenvalues, gap)
    sigma_A = ht.compute_sigma(R_A, T_A, F_re, F_ah)
    sigma_B = ht.compute_sigma(R_B, T_B, F_re, F_ah)
    tau = ht.spectral_transfer(sigma_A, sigma_B, P, R_A, R_B, F)
    return np.real(np.asarray(tau)).ravel(), expansion


def _pvh_tau(kpar, *, gap: float = GAP) -> np.ndarray:
    """Both polarizations of the PVH per-mode transmissivity at |k_par|."""
    kpar = np.asarray(kpar)
    kz0 = lifshitz._kz(1.0 + 0j, OMEGA, jnp.asarray(kpar))
    total = np.zeros_like(kpar, dtype=float)
    for pol in ("s", "p"):
        R_A, T_A = lifshitz.slab_RT(EPS_A, OMEGA, jnp.asarray(kpar), THICKNESS_A, pol)
        R_B, T_B = lifshitz.slab_RT(EPS_B, OMEGA, jnp.asarray(kpar), THICKNESS_B, pol)
        total += np.asarray(
            lifshitz.polder_van_hove_per_mode(R_A, T_A, R_B, T_B, kz0, gap)
        )
    return total


# --------------------------------------------------------------------------
# propagating / evanescent split
# --------------------------------------------------------------------------


def _compare_region(region: str) -> tuple[float, float, float]:
    lo, hi = KPAR_PROP_RANGE if region == "prop" else KPAR_EVAN_RANGE
    kpar = np.linspace(lo, hi, N_KPAR_SPLIT) * OMEGA
    rcwa, _ = _rcwa_tau(_on_axis(kpar))
    pvh = _pvh_tau(kpar)
    pointwise = float(np.max(np.abs(rcwa - pvh) / np.abs(pvh)))
    int_rcwa = float(np.trapezoid(kpar * rcwa, kpar))  # radial measure k dk
    int_pvh = float(np.trapezoid(kpar * pvh, kpar))
    return pointwise, abs(int_rcwa - int_pvh) / abs(int_pvh), int_pvh


def test_propagating_channels_match_pvh():
    pointwise, integrated, _ = _compare_region("prop")
    assert pointwise < SPLIT_TOL, f"propagating pointwise err {pointwise:.3e}"
    assert integrated < SPLIT_TOL, f"propagating integrated err {integrated:.3e}"


def test_evanescent_channels_match_pvh():
    """Direct test of the 1/(2|k_z|) evanescent normalization."""
    pointwise, integrated, _ = _compare_region("evan")
    assert pointwise < SPLIT_TOL, f"evanescent pointwise err {pointwise:.3e}"
    assert integrated < SPLIT_TOL, f"evanescent integrated err {integrated:.3e}"


# --------------------------------------------------------------------------
# spectral sweep across k_par, several gaps
# --------------------------------------------------------------------------


def test_spectral_agreement_across_kpar():
    """Pointwise over a log grid of k_par at three gaps; also writes a plot."""
    k = np.geomspace(KPAR_MIN, KPAR_MAX, N_KPAR_SPECTRAL)
    kpar = k[np.abs(k - 1.0) > LIGHTLINE_EXCLUSION] * OMEGA

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    worst = {}
    for gap in SPECTRAL_GAPS:
        rcwa, _ = _rcwa_tau(_on_axis(kpar), gap=gap)
        pvh = _pvh_tau(kpar, gap=gap)
        rel = np.abs(rcwa - pvh) / np.maximum(np.abs(pvh), 1e-300)
        worst[gap] = float(rel.max())
        ax1.loglog(kpar / OMEGA, pvh, label=f"PVH d={gap}")
        ax1.loglog(kpar / OMEGA, rcwa, ":", lw=2.5)
        ax2.loglog(kpar / OMEGA, rel, label=f"d={gap}")
    ax1.set_ylabel(r"$\tau(\omega, k_\parallel)$ (PVH solid, RCWA dotted)")
    ax1.axvline(1.0, color="gray", lw=0.5)
    ax1.legend(fontsize=8)
    ax2.set_xlabel(r"$k_\parallel c/\omega$")
    ax2.set_ylabel("relative error")
    ax2.axvline(1.0, color="gray", lw=0.5)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    for gap, err in worst.items():
        assert err < SPECTRAL_TOL, f"gap {gap}: max pointwise rel err {err:.3e}"


# --------------------------------------------------------------------------
# generic azimuth and the multi-order channel sum
# --------------------------------------------------------------------------


def test_generic_azimuth_matches_pvh():
    """Off the k_x axis, where the H-basis 2x2 blocks are not diagonal."""
    azimuths = np.deg2rad([0.0, 17.0, 31.0, 45.0, 63.0, 88.0])
    radii = np.array([0.3, 0.7, 1.3, 2.5, 6.0]) * OMEGA

    k_points = np.array(
        [[r * np.cos(a), r * np.sin(a)] for r in radii for a in azimuths]
    )
    kpar = np.linalg.norm(k_points, axis=-1)

    rcwa, _ = _rcwa_tau(k_points)
    pvh = _pvh_tau(kpar)

    rel = np.abs(rcwa - pvh) / np.abs(pvh)
    worst = int(np.argmax(rel))
    assert rel.max() < AZIMUTH_TOL, (
        f"worst azimuth mismatch at k={k_points[worst]} "
        f"(|k|/omega={kpar[worst] / OMEGA:.3f}): rcwa={rcwa[worst]:.6e}, "
        f"pvh={pvh[worst]:.6e}, rel={rel.max():.3e}"
    )

    for r in radii:
        same_r = np.isclose(kpar, r, rtol=1e-12)
        spread = np.ptp(pvh[same_r]) / np.abs(pvh[same_r]).max()
        assert spread < 1e-12, f"PVH reference is not azimuthally symmetric: {spread}"


def test_multi_order_planar_matches_pvh_channel_sum():
    """With many orders a uniform slab decouples the G channels, so the trace is
    the channel-by-channel PVH sum, evanescent channels included."""
    k = np.array([[0.37, 0.11]]) * OMEGA

    for num_terms in (9, 21, 49):
        rcwa, expansion = _rcwa_tau(k, num_terms=num_terms)

        coeffs = np.asarray(expansion.basis_coefficients)
        k_channels = k[0][None, :] + coeffs * (2.0 * np.pi / PITCH)  # square lattice
        kpar_channels = np.linalg.norm(k_channels, axis=-1)

        near_light_line = np.abs(kpar_channels / OMEGA - 1.0) < LIGHT_LINE_MARGIN
        assert not near_light_line.any(), (
            "channel sits on the light line; choose a different k_par: "
            f"{kpar_channels[near_light_line] / OMEGA}"
        )

        pvh_sum = float(np.sum(_pvh_tau(kpar_channels)))
        rel = abs(float(rcwa[0]) - pvh_sum) / abs(pvh_sum)
        n_evan = int(np.sum(kpar_channels > OMEGA))
        assert rel < MULTI_ORDER_TOL, (
            f"num_terms={num_terms} ({coeffs.shape[0]} orders, {n_evan} "
            f"evanescent): rcwa={float(rcwa[0]):.10e}, "
            f"pvh_channel_sum={pvh_sum:.10e}, rel={rel:.3e}"
        )

        pvh_zero_order = float(np.sum(_pvh_tau(kpar_channels[:1])))
        assert abs(pvh_sum - pvh_zero_order) > 1e-3 * abs(pvh_sum), (
            "higher orders contribute nothing; test has no teeth"
        )


# --------------------------------------------------------------------------
# z-asymmetric bilayer: the S-matrix block convention
# --------------------------------------------------------------------------


def _cascade(r1, t1, r2, t2):
    """Composite (R, T) of two z-symmetric planar bodies in contact, from side 1."""
    denom = 1.0 - r1 * r2
    return r1 + t1 * r2 * t1 / denom, t1 * t2 / denom


def _bilayer_RT(kpar, pol, *, eps_near, thick_near, eps_far, thick_far):
    """(R_gap_side, R_far_side, T) of a planar bilayer, from lifshitz."""
    r_near, t_near = lifshitz.slab_RT(
        eps_near, OMEGA, jnp.asarray(kpar), thick_near, pol
    )
    r_far, t_far = lifshitz.slab_RT(eps_far, OMEGA, jnp.asarray(kpar), thick_far, pol)
    R_gap, T_gap = _cascade(r_near, t_near, r_far, t_far)
    R_far, _ = _cascade(r_far, t_far, r_near, t_near)
    return R_gap, R_far, T_gap


def _bilayer_setup(kpar_values):
    plv, expansion, _ = ss.make_rcwa_setup(pitch=PITCH, approximate_num_terms=1)
    kw = dict(
        wavelength=jnp.asarray(WAVELENGTH),
        in_plane_wavevector=jnp.asarray(_on_axis(kpar_values)),
        primitive_lattice_vectors=plv,
        expansion=expansion,
    )
    vac = ss.eigensolve_uniform(**kw, permittivity=1.0 + 0.0j)
    solves = {
        name: ss.eigensolve_uniform(**kw, permittivity=eps)
        for name, eps in (
            ("A_near", EPS_A_NEAR),
            ("A_far", EPS_A_FAR),
            ("B_near", EPS_B_NEAR),
            ("B_far", EPS_B_FAR),
        )
    }
    return vac, solves


def test_planar_bilayer_matches_pvh():
    """Pointwise PVH agreement for z-asymmetric planar bodies."""
    kpar = np.concatenate(
        [
            np.linspace(0.05, 0.95, 12) * OMEGA,  # propagating
            np.linspace(1.05, 6.0, 14) * OMEGA,  # evanescent
        ]
    )
    vac, solves = _bilayer_setup(kpar)

    R_A, T_A, T_A_far = stack_blocks(
        vac,
        [solves["A_far"], solves["A_near"]],
        [jnp.asarray(THICK_A_FAR), jnp.asarray(THICK_A_NEAR)],
        is_body_A=True,
    )
    R_B, T_B, T_B_far = stack_blocks(
        vac,
        [solves["B_near"], solves["B_far"]],
        [jnp.asarray(THICK_B_NEAR), jnp.asarray(THICK_B_FAR)],
        is_body_A=False,
    )
    F_re, F_ah, F = ht.poynting_flux_matrices(vac)
    P = ht.propagation_matrix(vac.eigenvalues, GAP)
    sigma_A = ht.compute_sigma(R_A, T_A, F_re, F_ah)
    sigma_B = ht.compute_sigma(R_B, T_B, F_re, F_ah)
    rcwa = np.real(
        np.asarray(ht.spectral_transfer(sigma_A, sigma_B, P, R_A, R_B, F))
    ).ravel()

    kz0 = lifshitz._kz(1.0 + 0j, OMEGA, jnp.asarray(kpar))
    pvh = np.zeros_like(kpar)
    pvh_wrong_side = np.zeros_like(kpar)
    asymmetry = 0.0
    for pol in ("s", "p"):
        RA_gap, RA_far, TA = _bilayer_RT(
            kpar,
            pol,
            eps_near=EPS_A_NEAR,
            thick_near=THICK_A_NEAR,
            eps_far=EPS_A_FAR,
            thick_far=THICK_A_FAR,
        )
        RB_gap, RB_far, TB = _bilayer_RT(
            kpar,
            pol,
            eps_near=EPS_B_NEAR,
            thick_near=THICK_B_NEAR,
            eps_far=EPS_B_FAR,
            thick_far=THICK_B_FAR,
        )
        pvh += np.asarray(
            lifshitz.polder_van_hove_per_mode(RA_gap, TA, RB_gap, TB, kz0, GAP)
        )
        # Same formula with each body's far-side reflection: what the trace would
        # have to match if the S-matrix blocks were swapped.
        pvh_wrong_side += np.asarray(
            lifshitz.polder_van_hove_per_mode(RA_far, TA, RB_far, TB, kz0, GAP)
        )
        asymmetry = max(
            asymmetry,
            float(np.max(np.abs(RA_gap - RA_far) / np.abs(RA_gap))),
            float(np.max(np.abs(RB_gap - RB_far) / np.abs(RB_gap))),
        )

    assert asymmetry > MIN_ASYMMETRY, (
        f"bilayers are nearly z-symmetric (max |dR|/|R| = {asymmetry:.3e}); "
        "the block convention would not be exercised"
    )

    rel = np.abs(rcwa - pvh) / np.abs(pvh)
    worst = int(np.argmax(rel))
    assert rel.max() < BILAYER_TOL, (
        f"z-asymmetric planar bilayer vs PVH: worst at "
        f"|k|/omega={kpar[worst] / OMEGA:.3f}, rcwa={rcwa[worst]:.6e}, "
        f"pvh={pvh[worst]:.6e}, rel={rel.max():.3e}"
    )

    rel_wrong = np.max(np.abs(rcwa - pvh_wrong_side) / np.abs(pvh_wrong_side))
    assert rel_wrong > MIN_ASYMMETRY, (
        "the gap-side and far-side references agree, so this test could not "
        f"detect a swapped block ({rel_wrong:.3e})"
    )

    for tag, T_gap, T_far in (("A", T_A, T_A_far), ("B", T_B, T_B_far)):
        rel_T = float(jnp.max(jnp.abs(T_gap - T_far)) / jnp.max(jnp.abs(T_gap)))
        assert rel_T < BILAYER_TOL, (
            f"planar body {tag}: T should be direction-independent, got {rel_T:.3e}"
        )


# --------------------------------------------------------------------------
# truncation convergence at small gap
# --------------------------------------------------------------------------


def _rcwa_bz_sum(num_terms: int) -> float:
    """BZ-averaged transfer per unit area, identical slabs, small gap."""
    plv, expansion, ipw = ss.make_rcwa_setup(
        pitch=PITCH_BZ, approximate_num_terms=num_terms, brillouin_grid_shape=BZ_GRID
    )
    kw = dict(
        wavelength=jnp.asarray(WAVELENGTH),
        in_plane_wavevector=ipw,
        primitive_lattice_vectors=plv,
        expansion=expansion,
    )
    vac = ss.eigensolve_uniform(**kw, permittivity=1.0 + 0.0j)
    slab = ss.eigensolve_uniform(**kw, permittivity=EPS_CONVERGENCE)
    thickness = jnp.asarray(THICKNESS_CONVERGENCE)
    R_A, T_A, _ = ss.body_s_matrices(vac, slab, thickness, is_body_A=True)
    R_B, T_B, _ = ss.body_s_matrices(vac, slab, thickness, is_body_A=False)
    F_re, F_ah, F = ht.poynting_flux_matrices(vac)
    P = ht.propagation_matrix(vac.eigenvalues, CONVERGENCE_GAP)
    sigma_A = ht.compute_sigma(R_A, T_A, F_re, F_ah)
    sigma_B = ht.compute_sigma(R_B, T_B, F_re, F_ah)
    tau = ht.spectral_transfer(sigma_A, sigma_B, P, R_A, R_B, F)
    n_bz = BZ_GRID[0] * BZ_GRID[1]
    area = float(np.asarray(ss.cell_area(plv)))
    return float(np.sum(np.real(np.asarray(tau))) / n_bz / area)


def test_convergence_in_orders_at_small_gap():
    """Truncation error falls by orders of magnitude, down to the quadrature floor.

    Truncation error saturates by ~50 orders; what remains is quadrature error
    (finite BZ grid, plus the reference's own k integral), which more orders cannot
    improve.
    """
    reference = float(
        np.asarray(
            lifshitz.polder_van_hove_integrated(
                omega=OMEGA,
                eps_A=EPS_CONVERGENCE,
                thickness_A=THICKNESS_CONVERGENCE,
                eps_B=EPS_CONVERGENCE,
                thickness_B=THICKNESS_CONVERGENCE,
                gap=CONVERGENCE_GAP,
                kpar_max_factor=50.0,
                n_kpar=8000,
            )
        )
    )
    errors = [
        abs(_rcwa_bz_sum(terms) - reference) / abs(reference) for terms in TERMS_SWEEP
    ]

    for (coarse_n, coarse), (fine_n, fine) in zip(
        zip(TERMS_SWEEP, errors), zip(TERMS_SWEEP[1:], errors[1:])
    ):
        assert fine < coarse or fine < QUADRATURE_FLOOR, (
            f"error grew from {coarse:.3e} at {coarse_n} terms to {fine:.3e} at "
            f"{fine_n} terms, and {fine:.3e} is above the {QUADRATURE_FLOOR:g} "
            f"quadrature floor, so this is a real loss of accuracy: {errors}"
        )
    assert errors[-1] < CONVERGENCE_FINAL_TOL, (
        f"finest truncation ({TERMS_SWEEP[-1]} terms) error {errors[-1]:.3e} "
        f"exceeds {CONVERGENCE_FINAL_TOL}; errors = {errors}"
    )
    assert errors[0] > 10 * QUADRATURE_FLOOR, (
        f"coarsest truncation ({TERMS_SWEEP[0]} terms) error {errors[0]:.3e} is "
        f"already near the {QUADRATURE_FLOOR:g} floor; the sweep has no teeth"
    )
