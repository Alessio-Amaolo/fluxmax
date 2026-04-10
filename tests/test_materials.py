import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fluxmax.materials import (
    ConstantPermittivity,
    complex_refractive_index,
    omega_range_nat,
    permittivity,
    resolve_material,
    wavelength_range_um,
)
from fluxmax.units import si_units


def _expected_scalar_epsilon(omega_nat: float, material: str) -> complex:
    _, medium = resolve_material(material)
    omega_min, omega_max = omega_range_nat(material)
    freq_range = medium.valid_freq_range
    if np.isclose(omega_nat, omega_min):
        frequency = float(freq_range.min)
    elif np.isclose(omega_nat, omega_max):
        frequency = float(freq_range.max)
    else:
        wavelength_um = float(
            si_units.omega_nat_to_wavelength_um(jnp.asarray(omega_nat))
        )
        frequency = 1.0 / wavelength_um
    eps_tensor = np.asarray(medium.epsilon(frequency), dtype=complex)
    return complex(np.diag(eps_tensor)[0])


def test_gold_permittivity_matches_meep_endpoints() -> None:
    omega_min, omega_max = omega_range_nat("gold")

    refractive_index = complex_refractive_index(
        jnp.asarray([omega_min, omega_max]), "gold"
    )
    expected = jnp.asarray(
        [
            jnp.sqrt(_expected_scalar_epsilon(omega_min, "gold")),
            jnp.sqrt(_expected_scalar_epsilon(omega_max, "gold")),
        ]
    )

    assert resolve_material("gold")[0] == "Au"
    assert jnp.allclose(refractive_index**2, expected**2)
    assert jnp.allclose(
        permittivity(jnp.asarray([omega_min]), "gold"), expected[:1] ** 2
    )


def test_gold_permittivity_rejects_extrapolation() -> None:
    _, omega_max = omega_range_nat("gold")
    omega_above_range = omega_max * 1.01

    with pytest.raises(ValueError, match="outside the valid meep range"):
        permittivity(jnp.asarray([omega_above_range]), "gold")


def test_gold_dispersion_diagnostic_plot() -> None:
    min_wavelength_um, max_wavelength_um = wavelength_range_um("gold")

    wavelength_um = jnp.linspace(min_wavelength_um, max_wavelength_um, 600)
    omega_nat_from_wavelength = si_units.wavelength_um_to_omega_nat(wavelength_um)
    epsilon_wavelength = permittivity(omega_nat_from_wavelength, "gold")

    omega_min = jnp.min(omega_nat_from_wavelength)
    omega_max = jnp.max(omega_nat_from_wavelength)
    omega_nat = jnp.linspace(omega_min, omega_max, 600)
    epsilon_omega = permittivity(omega_nat, "gold")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].plot(
        wavelength_um,
        jnp.real(epsilon_wavelength),
        label="Re(epsilon)",
        color="tab:blue",
    )
    axes[0].plot(
        wavelength_um,
        jnp.imag(epsilon_wavelength),
        label="Im(epsilon)",
        color="tab:orange",
    )
    axes[0].set_xlabel("wavelength [um]")
    axes[0].set_ylabel("permittivity")
    axes[0].set_title("Gold permittivity vs wavelength")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        omega_nat,
        jnp.real(epsilon_omega),
        label="Re(epsilon)",
        color="tab:blue",
    )
    axes[1].plot(
        omega_nat,
        jnp.imag(epsilon_omega),
        label="Im(epsilon)",
        color="tab:orange",
    )
    axes[1].set_xlabel("omega [1/L0]")
    axes[1].set_ylabel("permittivity")
    axes[1].set_title("Gold permittivity vs natural omega")
    axes[1].grid(True, alpha=0.3)

    output_path = "tests/test_output/gold_dispersion_diagnostic.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def test_constant_permittivity_broadcasts_and_has_infinite_range() -> None:
    material = ConstantPermittivity(eps=2.5 + 0.1j, name="my_eps")
    resolved_name, resolved_model = resolve_material(material)
    assert resolved_name == "my_eps"
    assert resolved_model is material

    omega = jnp.asarray([0.1, 0.2, 0.3])
    eps = permittivity(omega, material)
    assert eps.shape == omega.shape
    assert jnp.allclose(eps, (2.5 + 0.1j) * jnp.ones_like(omega, dtype=complex))

    omega_min, omega_max = omega_range_nat(material)
    wavelength_min, wavelength_max = wavelength_range_um(material)
    assert omega_min == 0.0
    assert np.isinf(omega_max)
    assert wavelength_min == 0.0
    assert np.isinf(wavelength_max)


def test_scalar_numeric_material_selector_is_constant_eps() -> None:
    omega = jnp.asarray([0.12, 0.34])
    eps = permittivity(omega, 3.0 + 0.0j)
    assert jnp.allclose(eps, 3.0 + 0.0j)
