"""
Tests that the RCWA-based transfer trace for planar slabs matches the
Polder-Van Hove (PVH) result, which is exact for planar structures. The
PVH result is computed using the implementation in fluxmax.physics.lifshitz.

Mostly generates plots and diagnostics, but does check that for high enough
number of Fourier terms, the values agree.
"""
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import fmmax  # noqa: E402

import fluxmax.physics.lifshitz as lifshitz  # noqa: E402
from fluxmax.physics import heat_transfer as ht  # noqa: E402
from fluxmax.setup import two_body as ss  # noqa: E402

EPS_SLAB = 4.0 + 0.5j
WAVELENGTH = 1.0
SLAB_THICKNESS = 0.5
PITCH = 1.0
GAPS = [0.1, 0.2, 0.5, 1.0]

APPROXIMATE_NUM_TERMS_BY_GAP = {0.1: 200, 0.2: 50, 0.5: 50, 1.0: 50}
BZ_GRID_BY_GAP = {0.1: (3, 3), 0.2: (3, 3), 0.5: (5, 5), 1.0: (3, 3)}
TERMS_SWEEP = [1, 5, 10, 25]

RELATIVE_ERROR_TOLERANCE = 0.10
MAX_ESTIMATED_GB = 1.5
ESTIMATE_MULTIPLIER = 80.0

OUTPUT_DIR = Path(__file__).resolve().parent / "test_output"
PLOT_PATH = OUTPUT_DIR / "planar_validation_sweep.png"


def _estimated_peak_memory_gb(
    *, actual_num_terms: int, bz_grid: tuple[int, int]
) -> float:
    n = 2 * int(actual_num_terms)
    bz_points = int(bz_grid[0]) * int(bz_grid[1])
    bytes_per_complex128 = 16
    base_bytes = bz_points * (n * n) * bytes_per_complex128
    return float(base_bytes * ESTIMATE_MULTIPLIER / 1e9)


def _rcwa_transfer_for_gap(
    gap: float, num_terms: int, bz_grid: tuple[int, int]
) -> float:
    wavelength = jnp.asarray(WAVELENGTH)
    thickness = jnp.asarray(SLAB_THICKNESS)
    gap_d = jnp.asarray(gap)

    plv, expansion, in_plane_wavevector = ss.make_rcwa_setup(
        pitch=PITCH,
        approximate_num_terms=num_terms,
        brillouin_grid_shape=bz_grid,
    )

    eigensolve_kw = dict(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=plv,
        expansion=expansion,
    )

    vac_lsr = ss.eigensolve_uniform(**eigensolve_kw, permittivity=1.0 + 0j)
    slab_lsr = ss.eigensolve_uniform(**eigensolve_kw, permittivity=EPS_SLAB)

    reflection_a, transmission_a, _ = ss.body_s_matrices(
        vac_lsr, slab_lsr, thickness, is_body_A=True
    )
    reflection_b, transmission_b, _ = ss.body_s_matrices(
        vac_lsr, slab_lsr, thickness, is_body_A=False
    )

    flux_re, flux_ah, flux = ht.poynting_flux_matrices(vac_lsr)
    sigma_a = ht.compute_sigma(reflection_a, transmission_a, flux_re, flux_ah)
    sigma_b = ht.compute_sigma(reflection_b, transmission_b, flux_re, flux_ah)
    propagation = ht.propagation_matrix(vac_lsr.eigenvalues, gap_d)
    tau = ht.spectral_transfer(
        sigma_a,
        sigma_b,
        propagation,
        reflection_a,
        reflection_b,
        flux,
    )
    n_bz = bz_grid[0] * bz_grid[1]
    area = ss.cell_area(plv)
    return float(jnp.sum(jnp.real(tau)) / n_bz / area)


def _pvh_transfer_for_gap(gap: float) -> float:
    omega = fmmax.angular_frequency_for_wavelength(  # type: ignore[attr-defined]
        jnp.asarray(WAVELENGTH)
    )
    return float(
        lifshitz.polder_van_hove_integrated(
            omega=omega,
            eps_A=EPS_SLAB,
            thickness_A=SLAB_THICKNESS,
            eps_B=EPS_SLAB,
            thickness_B=SLAB_THICKNESS,
            gap=gap,
            kpar_max_factor=50.0,
            n_kpar=8000,
        )
    )


def _save_outputs(
    *, gaps: np.ndarray, pvh: np.ndarray, rcwa_by_terms: dict[int, np.ndarray]
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    terms_sorted = sorted(rcwa_by_terms.keys())
    if terms_sorted:
        rcwa = np.stack([rcwa_by_terms[terms] for terms in terms_sorted], axis=0)
    else:
        rcwa = np.zeros((0, gaps.size), dtype=float)

    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.semilogy(gaps, pvh, "k:", linewidth=2.0, label="PVH")

    if terms_sorted:
        cmap = plt.get_cmap("viridis", max(len(terms_sorted), 2))
        for index, terms in enumerate(terms_sorted):
            axis.semilogy(
                gaps,
                rcwa[index, :],
                "o-",
                color=cmap(index),
                markersize=4,
                linewidth=1.5,
                label=f"RCWA approx_terms={terms}",
            )

    axis.set_xlabel("Gap distance d / lambda")
    axis.set_ylabel("Transfer")
    axis.set_title("Planar slabs: PVH vs RCWA trace")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    figure.savefig(PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def test_planar_trace_matches_pvh_and_writes_diagnostics() -> None:
    gaps = np.asarray(GAPS, dtype=float)
    pvh = np.zeros_like(gaps)

    for index, gap in enumerate(gaps):
        pvh[index] = _pvh_transfer_for_gap(float(gap))

        num_terms_default = APPROXIMATE_NUM_TERMS_BY_GAP[float(gap)]
        bz_grid = BZ_GRID_BY_GAP[float(gap)]
        rcwa_default = _rcwa_transfer_for_gap(float(gap), num_terms_default, bz_grid)

        if abs(pvh[index]) > 1e-30:
            relative_error = abs(rcwa_default - pvh[index]) / abs(pvh[index])
        else:
            relative_error = abs(rcwa_default - pvh[index])

        assert relative_error < RELATIVE_ERROR_TOLERANCE, (
            f"gap={gap:.2f}: rcwa={rcwa_default:.6e}, pvh={pvh[index]:.6e}, "
            f"rel_err={relative_error:.3e}"
        )

    rcwa_by_terms: dict[int, np.ndarray] = {}
    unique_bz_grids = [
        cast(tuple[int, int], grid) for grid in sorted(set(BZ_GRID_BY_GAP.values()))
    ]
    for terms in TERMS_SWEEP:
        worst_estimated_gb = max(
            _estimated_peak_memory_gb(actual_num_terms=terms, bz_grid=bz_grid)
            for bz_grid in unique_bz_grids
        )
        if worst_estimated_gb > MAX_ESTIMATED_GB:
            continue

        values = np.zeros_like(gaps)
        for index, gap in enumerate(gaps):
            bz_grid = BZ_GRID_BY_GAP[float(gap)]
            values[index] = _rcwa_transfer_for_gap(float(gap), int(terms), bz_grid)
        rcwa_by_terms[int(terms)] = values

    _save_outputs(gaps=gaps, pvh=pvh, rcwa_by_terms=rcwa_by_terms)
