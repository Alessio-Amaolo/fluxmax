"""Dispersive material utilities backed by tabulated optical data."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import jax.numpy as jnp
import numpy as np

from fluxmax.units import si_units


@dataclass(frozen=True)
class MaterialOpticalData:
    """Tabulated optical constants for a material.

    Parameters
    ----------
    source
        Human-readable citation from the first line of the data file.
    wavelength_n_um
        Wavelength samples for the refractive-index table in microns.
    n_values
        Refractive index values sampled at ``wavelength_n_um``.
    wavelength_k_um
        Wavelength samples for the extinction-coefficient table in microns.
    k_values
        Extinction coefficient values sampled at ``wavelength_k_um``.
    """

    source: str
    wavelength_n_um: jnp.ndarray
    n_values: jnp.ndarray
    wavelength_k_um: jnp.ndarray
    k_values: jnp.ndarray

    @property
    def min_wavelength_um(self) -> float:
        """Lower interpolation bound shared by both tables."""
        return float(
            max(
                float(jnp.min(self.wavelength_n_um)),
                float(jnp.min(self.wavelength_k_um)),
            )
        )

    @property
    def max_wavelength_um(self) -> float:
        """Upper interpolation bound shared by both tables."""
        return float(
            min(
                float(jnp.max(self.wavelength_n_um)),
                float(jnp.max(self.wavelength_k_um)),
            )
        )


def available_materials() -> tuple[str, ...]:
    """Return the names of tabulated materials available in ``data``."""
    data_dir = files("fluxmax.materials").joinpath("data")
    names = []
    for path in data_dir.iterdir():
        name = path.name
        if name.endswith(".txt"):
            names.append(name.removesuffix(".txt"))
    return tuple(sorted(names))


@lru_cache(maxsize=None)
def load_material_data(material: str) -> MaterialOpticalData:
    """Load tabulated optical data for a material.

    Parameters
    ----------
    material
        Material name matching ``materials/data/<material>.txt``.

    Returns
    -------
    MaterialOpticalData
        Parsed optical-constant tables.

    Raises
    ------
    ValueError
        If the material file is missing or malformed.
    """
    material_name = material.strip().lower()
    try:
        data_path = files("fluxmax.materials").joinpath("data", f"{material_name}.txt")
    except ModuleNotFoundError as exc:
        raise ValueError("The fluxmax.materials package is not importable.") from exc

    if not data_path.is_file():
        available = ", ".join(available_materials()) or "none"
        raise ValueError(
            f"Unknown material '{material}'. Available materials: {available}."
        )

    text = data_path.read_text(encoding="utf-8")
    sections = [
        section.splitlines() for section in text.split("\n\n") if section.strip()
    ]
    if len(sections) != 3:
        raise ValueError(
            "Expected source + n-table + k-table in "
            f"{data_path.name}, found {len(sections)} sections."
        )

    source_section, n_section, k_section = sections
    if len(source_section) != 1:
        raise ValueError(f"Expected a one-line source section in {data_path.name}.")

    wavelength_n_um, n_values = _parse_table(n_section, expected_header=("wl", "n"))
    wavelength_k_um, k_values = _parse_table(k_section, expected_header=("wl", "k"))

    return MaterialOpticalData(
        source=source_section[0].strip(),
        wavelength_n_um=jnp.asarray(wavelength_n_um),
        n_values=jnp.asarray(n_values),
        wavelength_k_um=jnp.asarray(wavelength_k_um),
        k_values=jnp.asarray(k_values),
    )


def complex_refractive_index(
    omega_nat: jnp.ndarray,
    material: str,
    *,
    L0_m: float = si_units.L0_M_DEFAULT,
) -> jnp.ndarray:
    """Interpolate the complex refractive index ``n + i k`` at natural-unit frequencies.

    Parameters
    ----------
    omega_nat
        Angular frequencies in natural units ``1 / L0``.
    material
        Material name matching a file in ``materials/data``.
    L0_m
        Physical size of one natural length unit in meters.

    Returns
    -------
    jnp.ndarray
        Complex refractive index values with the same shape as ``omega_nat``.

    Raises
    ------
    ValueError
        If any frequency is non-positive or would require extrapolation.
    """
    data = load_material_data(material)
    wavelength_um = si_units.omega_nat_to_wavelength_um(omega_nat, L0_m=L0_m)
    min_wavelength = data.min_wavelength_um
    max_wavelength = data.max_wavelength_um
    min_omega = float(
        si_units.wavelength_um_to_omega_nat(jnp.asarray(max_wavelength), L0_m=L0_m)
    )
    max_omega = float(
        si_units.wavelength_um_to_omega_nat(jnp.asarray(min_wavelength), L0_m=L0_m)
    )

    wavelength_np = np.asarray(wavelength_um, dtype=float)
    if np.any(~np.isfinite(wavelength_np)):
        raise ValueError("omega_nat must contain only finite values.")
    if np.any(wavelength_np <= 0.0):
        raise ValueError("omega_nat must be strictly positive.")
    if np.any(wavelength_np < min_wavelength) or np.any(wavelength_np > max_wavelength):
        raise ValueError(
            "omega_nat is outside the tabulated range for "
            f"{material}. Supported omega range is [{min_omega}, {max_omega}] "
            "in natural units "
            f"for L0_m={L0_m}, equivalent to wavelength range "
            f"[{min_wavelength}, {max_wavelength}] microns."
        )

    n_interp = np.interp(
        wavelength_np.reshape(-1),
        np.asarray(data.wavelength_n_um, dtype=float),
        np.asarray(data.n_values, dtype=float),
    ).reshape(wavelength_np.shape)
    k_interp = np.interp(
        wavelength_np.reshape(-1),
        np.asarray(data.wavelength_k_um, dtype=float),
        np.asarray(data.k_values, dtype=float),
    ).reshape(wavelength_np.shape)
    return jnp.asarray(n_interp + 1j * k_interp)


def permittivity(
    omega_nat: jnp.ndarray,
    material: str,
    *,
    L0_m: float = si_units.L0_M_DEFAULT,
) -> jnp.ndarray:
    """Interpolate the dielectric function from tabulated ``n`` and ``k`` data.

    Parameters
    ----------
    omega_nat
        Angular frequencies in natural units ``1 / L0``.
    material
        Material name matching a file in ``materials/data``.
    L0_m
        Physical size of one natural length unit in meters.

    Returns
    -------
    jnp.ndarray
        Complex permittivity values ``(n + i k)^2`` with the same shape as
        ``omega_nat``.
    """
    refractive_index = complex_refractive_index(omega_nat, material, L0_m=L0_m)
    return refractive_index**2


def _parse_table(
    lines: list[str], *, expected_header: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a two-column wavelength table."""
    if len(lines) < 2:
        raise ValueError(
            "Optical-constant table must include a header and at least one row."
        )

    header = tuple(lines[0].split())
    if header != expected_header:
        raise ValueError(f"Expected header {expected_header}, found {header}.")

    wavelengths: list[float] = []
    values: list[float] = []
    for line in lines[1:]:
        wl_text, value_text = line.split()
        wavelengths.append(float(wl_text))
        values.append(float(value_text))

    wavelength_array = np.asarray(wavelengths, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if not np.all(np.diff(wavelength_array) > 0.0):
        raise ValueError(
            "Wavelength values must be strictly increasing for interpolation."
        )
    return wavelength_array, value_array
