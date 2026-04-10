"""Meep-backed material dispersion utilities."""

from .meep_materials import (
    ConstantPermittivity,
    MATERIAL_ALIASES,
    available_materials,
    complex_refractive_index,
    meep_frequency_from_omega_nat,
    omega_range_nat,
    permittivity,
    resolve_material,
    wavelength_range_um,
)
from .resonances import make_resonance_aware_omega_grid, real_epsilon_crossings

__all__ = [
    "ConstantPermittivity",
    "MATERIAL_ALIASES",
    "available_materials",
    "complex_refractive_index",
    "meep_frequency_from_omega_nat",
    "omega_range_nat",
    "permittivity",
    "resolve_material",
    "wavelength_range_um",
    "real_epsilon_crossings",
    "make_resonance_aware_omega_grid",
]
