"""Resonance-aware frequency-grid construction utilities.

This module provides helpers used by study scripts to detect approximate
resonance locations from material dispersion and to enrich omega grids near
those locations.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import warnings

from fluxmax.units import si_units

from . import meep_materials


def real_epsilon_crossings(
    material: meep_materials.MaterialSpec,
    *,
    target_values: tuple[float, ...],
    omega_range_nat: tuple[float, float],
    resonance_scan_points: int,
    L0_m: float = 100.0e-9,
) -> list[dict[str, float]]:
    """Estimate frequencies where ``Re(epsilon)`` crosses selected targets.

    The crossings are found by linear interpolation on a coarse uniform scan
    in omega.

    Parameters
    ----------
    material : MaterialSpec
        Material selector accepted by ``fluxmax.materials.resolve_material``.
    target_values : tuple of float
        Target values for ``Re(epsilon)`` (for example ``(-1.0, -2.0)``).
    omega_range_nat : tuple of float
        Requested scan range ``(omega_min, omega_max)`` in natural units.
    resonance_scan_points : int
        Number of scan points in the uniform omega grid.
    L0_m : float, optional
        Natural-unit length scale in meters.

    Returns
    -------
    list of dict
        Sorted list of crossing records. Each record contains keys
        ``"target"``, ``"omega"``, and ``"wavelength_um"``.

    Raises
    ------
    ValueError
        If the requested omega range does not overlap the material data range.
    """
    _, medium = meep_materials.resolve_material(material)
    medium_omega_min, medium_omega_max = meep_materials.omega_range_nat(medium, L0_m=L0_m)
    omega_min = max(float(omega_range_nat[0]), float(medium_omega_min))
    omega_max = min(float(omega_range_nat[1]), float(medium_omega_max))
    if omega_min >= omega_max:
        raise ValueError(
            "Requested omega range does not overlap the material dispersion range."
        )

    omega_nat = np.linspace(omega_min, omega_max, resonance_scan_points)
    eps_values = np.asarray(
        meep_materials.permittivity(jnp.asarray(omega_nat), medium, L0_m=L0_m),
        dtype=complex,
    )
    re_eps = np.real(eps_values)

    crossings: list[dict[str, float]] = []
    for target in target_values:
        shifted = re_eps - float(target)
        for idx in range(len(shifted) - 1):
            left = shifted[idx]
            right = shifted[idx + 1]
            if left == 0.0 or left * right < 0.0:
                frac = left / (left - right)
                omega_cross = omega_nat[idx] + frac * (
                    omega_nat[idx + 1] - omega_nat[idx]
                )
                wavelength_um_cross = float(
                    si_units.omega_nat_to_wavelength_um(
                        jnp.asarray(omega_cross), L0_m=L0_m
                    )
                )
                crossings.append(
                    {
                        "target": float(target),
                        "omega": float(omega_cross),
                        "wavelength_um": wavelength_um_cross,
                    }
                )

    crossings.sort(key=lambda item: item["omega"])
    return crossings


def make_resonance_aware_omega_grid(
    material: meep_materials.MaterialSpec,
    *,
    omega_study_range: tuple[float, float],
    num_base_omegas: int,
    resonance_window_um: jnp.ndarray,
    target_values: tuple[float, ...],
    resonance_scan_points: int,
    L0_m: float = 100.0e-9,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build an omega grid enriched near resonance crossing wavelengths.

    Parameters
    ----------
    material : MaterialSpec
        Material selector accepted by ``fluxmax.materials.resolve_material``.
    omega_study_range : tuple of float
        Requested omega range ``(omega_min, omega_max)`` in natural units.
    num_base_omegas : int
        Number of uniformly spaced base omega points.
    resonance_window_um : jax.Array
        Wavelength offsets (in microns) added around each crossing center.
    target_values : tuple of float
        Target values for ``Re(epsilon)`` crossing detection.
    resonance_scan_points : int
        Number of scan points for crossing detection.
    L0_m : float, optional
        Natural-unit length scale in meters.

    Returns
    -------
    tuple of jax.Array
        ``(omega_grid, wavelength_grid_um)`` sorted by increasing omega.

    Raises
    ------
    ValueError
        If the requested omega range does not overlap the material data range.
    """
    _, medium = meep_materials.resolve_material(material)
    wavelength_min_um, wavelength_max_um = meep_materials.wavelength_range_um(medium)
    medium_omega_min, medium_omega_max = meep_materials.omega_range_nat(medium, L0_m=L0_m)

    omega_min = max(float(omega_study_range[0]), float(medium_omega_min))
    omega_max = min(float(omega_study_range[1]), float(medium_omega_max))
    if omega_min >= omega_max:
        raise ValueError(
            "Requested omega range does not overlap the material dispersion range."
        )

    base_omegas = np.linspace(omega_min, omega_max, num_base_omegas)
    base_wavelengths_um = np.asarray(
        si_units.omega_nat_to_wavelength_um(jnp.asarray(base_omegas), L0_m=L0_m)
    )
    sampled_wavelengths = [
        float(value) for value in np.asarray(base_wavelengths_um, dtype=float)
    ]

    crossings = real_epsilon_crossings(
        material,
        target_values=target_values,
        omega_range_nat=(omega_min, omega_max),
        resonance_scan_points=resonance_scan_points,
        L0_m=L0_m,
    )
    for crossing in crossings:
        center = crossing["wavelength_um"]
        for offset in np.asarray(resonance_window_um, dtype=float):
            candidate = center + float(offset)
            if wavelength_min_um <= candidate <= wavelength_max_um:
                sampled_wavelengths.append(candidate)

    wavelength_um = np.asarray(
        sorted({round(value, 6) for value in sampled_wavelengths}), dtype=float
    )
    omega = np.asarray(
        si_units.wavelength_um_to_omega_nat(jnp.asarray(wavelength_um), L0_m=L0_m),
        dtype=float,
    )
    valid = (omega >= omega_min) & (omega <= omega_max)
    omega = omega[valid]
    wavelength_um = wavelength_um[valid]
    order = np.argsort(omega)

    return jnp.asarray(omega[order]), jnp.asarray(wavelength_um[order])
