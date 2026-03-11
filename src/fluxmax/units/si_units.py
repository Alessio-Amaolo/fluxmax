"""SI to natural-unit conversions for rcwa_id.

Convention
----------
The core code uses natural units with

- c = 1, hbar = 1, k_B = 1
- All lengths expressed in units of an arbitrary base length L0.

To connect to SI units, you must choose the physical size of L0.
The project default is:

- L0 = 100 nm

With that choice, all geometry specified as dimensionless numbers in the core
(e.g. a gap of 0.2) corresponds to 0.2 × 100 nm in physical units.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped

# Physical constants (SI)
HBAR_SI = 1.054571817e-34  # [J·s]
KB_SI = 1.380649e-23  # [J/K]
C_SI = 2.99792458e8  # [m/s]

# Default length unit: 1 natural length unit = 100 nm
L0_M_DEFAULT = 100e-9


@jaxtyped(typechecker=None)
def length_nat_to_m(
    x_nat: Float[Array, "*shape"] | float, *, L0_m: float = L0_M_DEFAULT
) -> Float[Array, "*shape"]:
    """Convert a length from natural units to meters."""
    return jnp.asarray(L0_m) * x_nat


@jaxtyped(typechecker=None)
def length_m_to_nat(
    x_m: Float[Array, "*shape"] | float, *, L0_m: float = L0_M_DEFAULT
) -> Float[Array, "*shape"]:
    """Convert a length in meters to natural units."""
    return x_m / jnp.asarray(L0_m)


@jaxtyped(typechecker=None)
def area_nat_to_m2(
    a_nat: Float[Array, "*shape"] | float, *, L0_m: float = L0_M_DEFAULT
) -> Float[Array, "*shape"]:
    """Convert an area from natural units (L0^2) to m^2."""
    return (jnp.asarray(L0_m) ** 2) * a_nat


@jaxtyped(typechecker=None)
def area_m2_to_nat(
    a_m2: Float[Array, "*shape"] | float, *, L0_m: float = L0_M_DEFAULT
) -> Float[Array, "*shape"]:
    """Convert an area in m^2 to natural units (L0^2)."""
    return a_m2 / (jnp.asarray(L0_m) ** 2)


@jaxtyped(typechecker=None)
def omega_phys_to_nat(
    omega_phys: Float[Array, "*shape"] | float, *, L0_m: float = L0_M_DEFAULT
) -> Float[Array, "*shape"]:
    """Convert angular frequency [rad/s] to natural units [1/L0].

    With c = 1 internally, omega is measured in inverse-length units.
    """
    return (jnp.asarray(L0_m) / C_SI) * omega_phys


@jaxtyped(typechecker=None)
def omega_nat_to_phys(
    omega_nat: Float[Array, "*shape"] | float, *, L0_m: float = L0_M_DEFAULT
) -> Float[Array, "*shape"]:
    """Convert natural angular frequency [1/L0] to SI [rad/s]."""
    return (C_SI / jnp.asarray(L0_m)) * omega_nat


@jaxtyped(typechecker=None)
def omega_nat_to_wavelength_um(
    omega_nat: Float[Array, "*shape"] | float, *, L0_m: float = L0_M_DEFAULT
) -> Float[Array, "*shape"]:
    """Convert natural-unit angular frequency to wavelength in microns."""
    omega_nat_array = jnp.asarray(omega_nat)
    wavelength_nat = 2.0 * jnp.pi / omega_nat_array
    wavelength_m = length_nat_to_m(wavelength_nat, L0_m=L0_m)
    return wavelength_m * 1e6


@jaxtyped(typechecker=None)
def wavelength_um_to_omega_nat(
    wavelength_um: Float[Array, "*shape"] | float, *, L0_m: float = L0_M_DEFAULT
) -> Float[Array, "*shape"]:
    """Convert wavelength in microns to natural-unit angular frequency."""
    wavelength_nat = length_m_to_nat(jnp.asarray(wavelength_um) * 1e-6, L0_m=L0_m)
    return 2.0 * jnp.pi / wavelength_nat


@jaxtyped(typechecker=None)
def temperature_K_to_nat(
    T_K: Float[Array, "*shape"] | float, *, L0_m: float = L0_M_DEFAULT
) -> Float[Array, "*shape"]:
    """Convert Kelvin to natural temperature units.

    In natural units k_B = 1, so temperature has the same units as energy.
    Using hbar = c = 1, that energy unit is hbar c / L0.
    """
    return (KB_SI * T_K * jnp.asarray(L0_m)) / (HBAR_SI * C_SI)


@jaxtyped(typechecker=None)
def temperature_nat_to_K(
    T_nat: Float[Array, "*shape"] | float, *, L0_m: float = L0_M_DEFAULT
) -> Float[Array, "*shape"]:
    """Convert natural temperature units to Kelvin."""
    return (T_nat * (HBAR_SI * C_SI)) / (KB_SI * jnp.asarray(L0_m))


@jaxtyped(typechecker=None)
def spectral_flux_density_nat_to_SI(
    phi_per_omega_per_area_nat: Float[Array, "*shape"] | float,
    *,
    L0_m: float = L0_M_DEFAULT,
) -> Float[Array, "*shape"]:
    """Convert natural spectral flux density to SI.

    Converts the quantity (1/A) dΦ/dω.

    Natural units: energy per area per angular-frequency.
    SI units: J/m^2 (equivalently W·s/m^2) per (rad/s).

    Scaling:
        (1/A dΦ/dω)_SI = (ħ c / L0^3) * (1/A dΦ/dω)_nat
    """
    return (HBAR_SI * C_SI / (jnp.asarray(L0_m) ** 3)) * phi_per_omega_per_area_nat


@jaxtyped(typechecker=None)
def flux_per_area_nat_to_SI(
    phi_per_area_nat: Float[Array, "*shape"] | float,
    *,
    L0_m: float = L0_M_DEFAULT,
) -> Float[Array, "*shape"]:
    """Convert total flux per area from natural units to SI [W/m^2].

    This is for quantities where the ω-integral has already been carried out
    in natural units.

    Scaling:
        (Φ/A)_SI = (ħ c^2 / L0^4) * (Φ/A)_nat
    """
    return (HBAR_SI * C_SI**2 / (jnp.asarray(L0_m) ** 4)) * phi_per_area_nat
