"""Material dispersion utilities."""

from .materials import (
    MaterialOpticalData,
    available_materials,
    complex_refractive_index,
    load_material_data,
    permittivity,
)

__all__ = [
    "MaterialOpticalData",
    "available_materials",
    "complex_refractive_index",
    "load_material_data",
    "permittivity",
]