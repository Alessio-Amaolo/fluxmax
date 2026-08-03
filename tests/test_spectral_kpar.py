"""Spectral comparison of tau(omega, k_par) vs Polder-Van Hove at fixed omega,
pointwise across k_par with relative error, for several gaps. Also saves a
diagnostic plot of the spectra and relative errors to tests/test_output/.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fluxmax.physics.lifshitz as lifshitz
from fluxmax.physics import heat_transfer as ht
from fluxmax.setup import two_body as ss

jax.config.update("jax_enable_x64", True)

WAVELENGTH = 1.0
PITCH = 1.0
EPS_A = 4.0 + 0.5j
EPS_B = 6.0 + 1.0j
THICKNESS_A = 0.5
THICKNESS_B = 0.35
GAPS = [0.05, 0.2, 1.0]
OMEGA = 2.0 * np.pi / WAVELENGTH

N_KPAR = 200
KPAR_MIN, KPAR_MAX = 0.02, 10.0  # units of omega/c
LIGHTLINE_EXCLUSION = 0.02  # skip +-2% around |k| = omega/c
POINTWISE_TOL = 1e-6

OUTPUT_DIR = Path(__file__).resolve().parent / "test_output"
PLOT_PATH = OUTPUT_DIR / "spectral_kpar_comparison.png"


def _kpar_grid() -> np.ndarray:
    k = np.geomspace(KPAR_MIN, KPAR_MAX, N_KPAR)
    return k[np.abs(k - 1.0) > LIGHTLINE_EXCLUSION] * OMEGA


def _rcwa_tau(kpar: np.ndarray, gap: float) -> np.ndarray:
    plv, expansion, _ = ss.make_rcwa_setup(pitch=PITCH, approximate_num_terms=1)
    ipw = jnp.stack([jnp.asarray(kpar), jnp.zeros_like(jnp.asarray(kpar))], axis=-1)
    kw = dict(
        wavelength=jnp.asarray(WAVELENGTH),
        in_plane_wavevector=ipw,
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
    return np.real(
        np.asarray(ht.spectral_transfer(sigma_A, sigma_B, P, R_A, R_B, F))
    ).ravel()


def _pvh_tau(kpar: np.ndarray, gap: float) -> np.ndarray:
    kz0 = lifshitz._kz(1.0 + 0j, OMEGA, jnp.asarray(kpar))
    total = np.zeros_like(kpar, dtype=float)
    for pol in ("s", "p"):
        R_A, T_A = lifshitz.slab_RT(EPS_A, OMEGA, jnp.asarray(kpar), THICKNESS_A, pol)
        R_B, T_B = lifshitz.slab_RT(EPS_B, OMEGA, jnp.asarray(kpar), THICKNESS_B, pol)
        total += np.asarray(
            lifshitz.polder_van_hove_per_mode(R_A, T_A, R_B, T_B, kz0, gap)
        )
    return total


def test_spectral_agreement_across_kpar():
    kpar = _kpar_grid()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    worst = {}
    for gap in GAPS:
        rcwa = _rcwa_tau(kpar, gap)
        pvh = _pvh_tau(kpar, gap)
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
        assert err < POINTWISE_TOL, f"gap {gap}: max pointwise rel err {err:.3e}"
