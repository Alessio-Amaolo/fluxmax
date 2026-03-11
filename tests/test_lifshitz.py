import jax.numpy as jnp
import pytest

from fluxmax.physics import lifshitz

OMEGA = 2.0
GAP = 0.2
EPS_LOSSY = 4.0 + 0.5j
THICK_APPROX = 80.0


@pytest.mark.parametrize("pol", ["s", "p"])
def test_halfspace_rt_matches_single_interface(pol: str) -> None:
    kpar = 0.75

    reflection, transmission = lifshitz.slab_RT(
        EPS_LOSSY,
        OMEGA,
        kpar,
        None,
        pol,
    )

    kz0 = lifshitz._kz(1.0 + 0j, OMEGA, kpar)
    kzs = lifshitz._kz(EPS_LOSSY, OMEGA, kpar)
    expected_reflection, _ = lifshitz.fresnel_interface(
        1.0 + 0j, EPS_LOSSY, kz0, kzs, pol
    )

    assert jnp.allclose(reflection, expected_reflection)
    assert jnp.allclose(transmission, 0.0 + 0.0j)


@pytest.mark.parametrize("pol", ["s", "p"])
def test_thick_lossy_slab_converges_to_halfspace(pol: str) -> None:
    kpar = jnp.asarray([0.5, 3.0])

    reflection_half, transmission_half = lifshitz.slab_RT(
        EPS_LOSSY,
        OMEGA,
        kpar,
        None,
        pol,
    )
    reflection_thick, transmission_thick = lifshitz.slab_RT(
        EPS_LOSSY,
        OMEGA,
        kpar,
        THICK_APPROX,
        pol,
    )

    assert jnp.allclose(reflection_thick, reflection_half, rtol=1e-6, atol=1e-8)
    assert jnp.allclose(transmission_thick, transmission_half, rtol=1e-6, atol=1e-8)


def test_halfspace_integrated_transfer_matches_thick_lossy_slab() -> None:
    halfspace = float(
        lifshitz.polder_van_hove_integrated(
            omega=OMEGA,
            eps_A=EPS_LOSSY,
            thickness_A=None,
            eps_B=EPS_LOSSY,
            thickness_B=None,
            gap=GAP,
            kpar_max_factor=20.0,
            n_kpar=1200,
        )
    )
    thick = float(
        lifshitz.polder_van_hove_integrated(
            omega=OMEGA,
            eps_A=EPS_LOSSY,
            thickness_A=THICK_APPROX,
            eps_B=EPS_LOSSY,
            thickness_B=THICK_APPROX,
            gap=GAP,
            kpar_max_factor=20.0,
            n_kpar=1200,
        )
    )

    assert halfspace == pytest.approx(thick, rel=1e-4, abs=1e-7)


def test_halfspace_trace_integrated_transfer_matches_thick_lossy_slab() -> None:
    halfspace = float(
        lifshitz.integrated_transfer(
            omega=OMEGA,
            eps_A=EPS_LOSSY,
            thickness_A=None,
            eps_B=EPS_LOSSY,
            thickness_B=None,
            gap=GAP,
            kpar_max_factor=20.0,
            n_kpar=1200,
        )
    )
    thick = float(
        lifshitz.integrated_transfer(
            omega=OMEGA,
            eps_A=EPS_LOSSY,
            thickness_A=THICK_APPROX,
            eps_B=EPS_LOSSY,
            thickness_B=THICK_APPROX,
            gap=GAP,
            kpar_max_factor=20.0,
            n_kpar=1200,
        )
    )

    assert halfspace == pytest.approx(thick, rel=1e-4, abs=1e-7)
