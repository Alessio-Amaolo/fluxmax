"""Thin wrappers around ``meep.materials`` dispersion models."""

from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import TypeAlias

import jax.numpy as jnp
import meep as mp
import meep.materials as meep_materials
import numpy as np
import numpy.typing as npt

from fluxmax.units import si_units


@dataclass(frozen=True, slots=True)
class ConstantPermittivity:
    """Frequency-independent isotropic permittivity material.

    This is a lightweight, non-meep material spec that can be passed anywhere
    a meep material name/medium is accepted by this module.

    Parameters
    ----------
    eps
        Complex relative permittivity $\epsilon$.
    name
        Human-readable identifier used in plots/outputs.
    """

    eps: complex
    name: str = "constant_eps"


MaterialSpec: TypeAlias = str | mp.Medium | ConstantPermittivity | complex | float
ResolvedMaterial: TypeAlias = mp.Medium | ConstantPermittivity

MATERIAL_ALIASES = {
    "gold": "Au",
    "silver": "Ag",
    "copper": "Cu",
    "aluminum": "Al",
    "beryllium": "Be",
    "chromium": "Cr",
    "nickel": "Ni",
    "palladium": "Pd",
    "platinum": "Pt",
    "titanium": "Ti",
    "tungsten": "W",
    "silicon": "Si",
    "c-si": "cSi",
    "asi": "aSi",
    "a-si": "aSi",
    "a-si:h": "aSi_H",
    "sio2": "SiO2",
    "fused silica": "fused_quartz",
    "fused quartz": "fused_quartz",
}


def available_materials() -> tuple[str, ...]:
    """Return the names of built-in ``meep.materials`` media."""
    names = []
    for name, value in vars(meep_materials).items():
        if isinstance(value, mp.Medium):
            names.append(name)
    return tuple(sorted(names))


def resolve_material(material: MaterialSpec) -> tuple[str, mp.Medium]:
    """Resolve a material selector into a canonical material model.

    Supported selectors:

    - ``str``: looked up in ``meep.materials`` (with aliases)
    - ``mp.Medium``: passed through directly
    - ``ConstantPermittivity``: passed through directly
    - numeric (float/complex): interpreted as constant scalar permittivity
    """
    if isinstance(material, mp.Medium):
        return "custom_medium", material

    if isinstance(material, ConstantPermittivity):
        return material.name, material

    if isinstance(material, numbers.Number) and not isinstance(material, bool):
        eps_value = complex(material)
        return "constant_eps", ConstantPermittivity(eps=eps_value)

    canonical_name = MATERIAL_ALIASES.get(str(material).strip().lower(), str(material).strip())
    medium = getattr(meep_materials, canonical_name, None)
    if not isinstance(medium, mp.Medium):
        available = ", ".join(available_materials()) or "none"
        raise ValueError(
            f"Unknown meep material '{material}'. Available materials: {available}."
        )
    return canonical_name, medium


def meep_frequency_from_omega_nat(
    omega_nat: jnp.ndarray,
    *,
    L0_m: float = si_units.L0_M_DEFAULT,
) -> jnp.ndarray:
    """Convert natural-unit angular frequency to meep material-library frequency."""
    wavelength_um = si_units.omega_nat_to_wavelength_um(omega_nat, L0_m=L0_m)
    return 1.0 / wavelength_um


def wavelength_range_um(material: MaterialSpec) -> tuple[float, float]:
    """Return the valid wavelength interval in microns.

    For meep materials, this is taken from ``valid_freq_range``.
    For ``ConstantPermittivity``, this returns ``(0.0, +inf)``.
    """
    _, medium = resolve_material(material)
    if isinstance(medium, ConstantPermittivity):
        return 0.0, float("inf")

    freq_range = getattr(medium, "valid_freq_range", None)
    if freq_range is None:
        raise ValueError("The selected meep material does not expose valid_freq_range.")
    return 1.0 / float(freq_range.max), 1.0 / float(freq_range.min)


def _clip_frequency_to_valid_range(
    frequency: npt.NDArray[np.float64],
    medium: mp.Medium,
) -> npt.NDArray[np.float64]:
    """Snap roundoff-near endpoints onto meep's valid frequency interval."""
    freq_range = getattr(medium, "valid_freq_range", None)
    if freq_range is None:
        return frequency

    freq_min = float(freq_range.min)
    freq_max = float(freq_range.max)
    tolerance = 1e-12 * max(1.0, abs(freq_min), abs(freq_max))
    clipped_frequency = np.clip(frequency, freq_min - tolerance, freq_max + tolerance)
    bounded_frequency: npt.NDArray[np.float64] = np.asarray(
        np.clip(clipped_frequency, freq_min, freq_max), dtype=float
    )
    return bounded_frequency


def omega_range_nat(
    material: MaterialSpec,
    L0_m: float = si_units.L0_M_DEFAULT,
) -> tuple[float, float]:
    """Return the valid natural-unit omega interval.

    For ``ConstantPermittivity``, this returns ``(0.0, +inf)``.
    """
    _, medium = resolve_material(material)
    if isinstance(medium, ConstantPermittivity):
        return 0.0, float("inf")

    min_wavelength_um, max_wavelength_um = wavelength_range_um(material)
    return (
        float(
            si_units.wavelength_um_to_omega_nat(
                jnp.asarray(max_wavelength_um), L0_m=L0_m
            )
        ),
        float(
            si_units.wavelength_um_to_omega_nat(
                jnp.asarray(min_wavelength_um), L0_m=L0_m
            )
        ),
    )


def permittivity(
    omega_nat: jnp.ndarray,
    material: MaterialSpec,
    L0_m: float = si_units.L0_M_DEFAULT,
) -> jnp.ndarray:
    """Evaluate scalar complex permittivity.

    For meep materials, this evaluates the underlying dispersion model.
    For ``ConstantPermittivity`` (or a scalar numeric selector), this returns a
    broadcasted constant complex array.
    """
    material_name, medium = resolve_material(material)
    omega_array = jnp.asarray(omega_nat)
    omega_np = np.asarray(omega_array, dtype=float)

    if np.any(~np.isfinite(omega_np)):
        raise ValueError("omega_nat must contain only finite values.")
    if np.any(omega_np <= 0.0):
        raise ValueError("omega_nat must be strictly positive.")

    if isinstance(medium, ConstantPermittivity):
        return jnp.full(omega_array.shape, medium.eps, dtype=complex)

    min_omega, max_omega = omega_range_nat(medium, L0_m=L0_m)
    if np.any(omega_np < min_omega) or np.any(omega_np > max_omega):
        min_wavelength, max_wavelength = wavelength_range_um(medium)
        raise ValueError(
            "omega_nat is outside the valid meep range for "
            f"{material_name}. Supported omega range is [{min_omega}, {max_omega}] "
            "in natural units "
            f"for L0_m={L0_m}, equivalent to wavelength range "
            f"[{min_wavelength}, {max_wavelength}] microns."
        )

    meep_frequency = np.asarray(
        meep_frequency_from_omega_nat(omega_array, L0_m=L0_m),
        dtype=float,
    )
    meep_frequency = _clip_frequency_to_valid_range(meep_frequency, medium)
    eps_values: list[complex] = []

    for freq in meep_frequency.reshape(-1):
        eps_tensor = np.asarray(medium.epsilon(float(freq)), dtype=complex)
        diagonal = np.diag(eps_tensor)
        if not np.allclose(eps_tensor, np.diag(diagonal)):
            raise ValueError(
                "Only isotropic meep materials without off-diagonal epsilon "
                "terms are supported."
            )
        if not np.allclose(diagonal, diagonal[0]):
            raise ValueError(
                "Only isotropic meep materials with scalar permittivity are supported."
            )
        eps_values.append(diagonal[0])

    return jnp.asarray(np.asarray(eps_values, dtype=complex).reshape(omega_array.shape))


def complex_refractive_index(
    omega_nat: jnp.ndarray,
    material: MaterialSpec,
    *,
    L0_m: float = si_units.L0_M_DEFAULT,
) -> jnp.ndarray:
    """Return ``n + i k`` reconstructed from the meep complex permittivity."""
    epsilon = np.asarray(permittivity(omega_nat, material, L0_m=L0_m), dtype=complex)
    epsilon_abs = np.abs(epsilon)
    n_values = np.sqrt(np.maximum((epsilon_abs + epsilon.real) / 2.0, 0.0))
    k_values = np.sqrt(np.maximum((epsilon_abs - epsilon.real) / 2.0, 0.0))
    k_values = np.copysign(k_values, epsilon.imag)
    return jnp.asarray(n_values + 1j * k_values)
