"""Meep-backed material dispersion utilities."""

from .meep_materials import (
    MATERIAL_ALIASES,
    available_materials,
    complex_refractive_index,
    meep_frequency_from_omega_nat,
    omega_range_nat,
    permittivity,
    resolve_material,
    wavelength_range_um,
)

__all__ = [
    "MATERIAL_ALIASES",
    "available_materials",
    "complex_refractive_index",
    "meep_frequency_from_omega_nat",
    "omega_range_nat",
    "permittivity",
    "resolve_material",
    "wavelength_range_um",
]
