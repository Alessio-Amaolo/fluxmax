"""Differentiability of the transfer.

Checked here against central finite differences:

  * the design gradient d tau / d rho through the full chain
    (density -> projection -> permittivity -> eigensolve -> S-matrix -> Sigma ->
    trace), with ``Formulation.FFT``;
  * the geometry gradients d tau / d(gap) and d tau / d(thickness), which flow
    through the gap propagator and the layer S-matrices;
  * that nothing produces NaN, including for a uniform layer, whose eigenvalues
    are massively degenerate and for a lossless structure, where tau itself vanishes.

The last test pins a known limitation instead of a correctness property: fmmax's
vector formulations call ``jax.lax.stop_gradient`` on the permittivity before
computing the tangent vector field, so ``jax.grad`` returns the derivative at
fixed vector field rather than the derivative of the objective it evaluates.

Because of that, :func:`~fluxmax.setup.two_body.eigensolve_patterned` overrides
fmmax and defaults to ``Formulation.FFT`` (exact gradients, slower convergence
in the number of orders), so the whole kernel chain is differentiable by
default; the vector formulations remain available by passing ``formulation``
explicitly.

If fmmax ever drops the ``stop_gradient``, the last test fails and we can default
to a vector formulation again.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
from fmmax.fmm import Formulation  # type: ignore[attr-defined]

from fluxmax.optimization.design_tools import (
    metallic_eps_from_density,
    project_tanh,
)
from fluxmax.physics import heat_transfer as ht
from fluxmax.physics import kernels
from fluxmax.setup import two_body as ss

PITCH = 1.0
NX = 16
NUM_TERMS = 9
OMEGA = 2.0 * np.pi
K_POINT = jnp.asarray([0.37 * 2 * np.pi, 0.11 * 2 * np.pi])
EPS_SOLID = 12.0 + 0.5j
EPS_VOID = 1.0 + 0.0j
THICKNESS = 0.4
GAP = 0.3
BETA = 4.0

FD_STEP = 1e-6
GRADIENT_RTOL = 1e-5
GEOMETRY_FD_STEP = 1e-6
GEOMETRY_RTOL = 1e-5
MIN_JONES_DISAGREEMENT = 0.1

_PLV, _EXPANSION, _ = ss.make_rcwa_setup(pitch=PITCH, approximate_num_terms=NUM_TERMS)
_RNG = np.random.default_rng(0)
_RHO0 = jnp.asarray(_RNG.uniform(0.2, 0.8, size=(NX, NX)))


def _eps_from_density(rho, eps_solid=EPS_SOLID):
    return metallic_eps_from_density(project_tanh(rho, beta=BETA), eps_solid, EPS_VOID)


def _tau(
    rho,
    *,
    thickness=THICKNESS,
    gap=GAP,
    formulation=Formulation.FFT,
    eps_solid=EPS_SOLID,
):
    """Real part of the single-k transfer, built from the patterned slab."""
    kw = dict(
        wavelength=jnp.asarray(2.0 * jnp.pi / OMEGA),
        in_plane_wavevector=K_POINT[jnp.newaxis, :],
        primitive_lattice_vectors=_PLV,
        expansion=_EXPANSION,
    )
    vac = ss.eigensolve_uniform(**kw, permittivity=1.0 + 0.0j)
    slab = ss.eigensolve_patterned(
        **kw,
        permittivity_array=_eps_from_density(rho, eps_solid),
        formulation=formulation,
    )
    R_A, T_A, _ = ss.body_s_matrices(vac, slab, jnp.asarray(thickness), is_body_A=True)
    R_B, T_B, _ = ss.body_s_matrices(vac, slab, jnp.asarray(thickness), is_body_A=False)
    F_re, F_ah, F = ht.poynting_flux_matrices(vac)
    P = ht.propagation_matrix(vac.eigenvalues, jnp.asarray(gap))
    sigma_A = ht.compute_sigma(R_A, T_A, F_re, F_ah)
    sigma_B = ht.compute_sigma(R_B, T_B, F_re, F_ah)
    return jnp.real(jnp.sum(ht.spectral_transfer(sigma_A, sigma_B, P, R_A, R_B, F)))


def _unit_directions(count: int):
    out = []
    rng = np.random.default_rng(7)
    for _ in range(count):
        v = jnp.asarray(rng.normal(size=(NX, NX)))
        out.append(v / jnp.linalg.norm(v))
    return out


def _directional_fd(fn, point, direction, step):
    return float(
        (fn(point + step * direction) - fn(point - step * direction)) / (2 * step)
    )


def test_design_gradient_matches_finite_differences():
    """d tau / d rho through the whole differentiable chain."""

    def objective(rho):
        return _tau(rho, formulation=Formulation.FFT)

    grad = np.asarray(jax.grad(objective)(_RHO0))
    assert np.all(np.isfinite(grad)), "design gradient contains non-finite entries"
    assert np.linalg.norm(grad) > 0, "design gradient is identically zero"

    for index, direction in enumerate(_unit_directions(4)):
        ad = float(np.sum(grad * np.asarray(direction)))
        fd = _directional_fd(objective, _RHO0, direction, FD_STEP)
        rel = abs(ad - fd) / max(abs(fd), 1e-30)
        assert rel < GRADIENT_RTOL, (
            f"direction {index}: AD={ad:.10e} vs FD={fd:.10e}, rel={rel:.3e}"
        )


def test_default_formulation_gives_an_exact_design_gradient():
    """The default path must be differentiable, not just an opt-in one."""
    default = (
        inspect.signature(ss.eigensolve_patterned).parameters["formulation"].default
    )
    assert default is Formulation.FFT, (
        f"eigensolve_patterned defaults to {default} rather than Formulation.FFT; "
        "the design gradient of the default path is biased (see the module "
        "docstring)."
    )

    def objective(rho):
        return jnp.real(
            kernels.two_body_tau_kernel(
                omega=OMEGA,
                in_plane_wavevector=K_POINT,
                primitive_lattice_vectors=_PLV,
                expansion=_EXPANSION,
                slab_permittivity=_eps_from_density(rho),
                slab_thickness=THICKNESS,
                gap=GAP,
            )
        )

    grad = np.asarray(jax.grad(objective)(_RHO0))
    assert np.all(np.isfinite(grad)), "default-path design gradient is not finite"
    assert np.linalg.norm(grad) > 0, "default-path design gradient is identically zero"

    for index, direction in enumerate(_unit_directions(4)):
        ad = float(np.sum(grad * np.asarray(direction)))
        fd = _directional_fd(objective, _RHO0, direction, FD_STEP)
        rel = abs(ad - fd) / max(abs(fd), 1e-30)
        assert rel < GRADIENT_RTOL, (
            f"default formulation, direction {index}: AD={ad:.10e} vs "
            f"FD={fd:.10e}, rel={rel:.3e}"
        )


def test_geometry_gradients_match_finite_differences():
    """d tau / d(gap) and d tau / d(thickness), including through the layer
    S-matrices and the gap propagator."""
    for name, wrap, value in (
        ("gap", lambda x: _tau(_RHO0, gap=x), GAP),
        ("thickness", lambda x: _tau(_RHO0, thickness=x), THICKNESS),
    ):
        point = jnp.asarray(value)
        ad = float(jax.grad(wrap)(point))
        assert np.isfinite(ad), f"d tau / d {name} is not finite"
        fd = float(
            (wrap(point + GEOMETRY_FD_STEP) - wrap(point - GEOMETRY_FD_STEP))
            / (2 * GEOMETRY_FD_STEP)
        )
        rel = abs(ad - fd) / max(abs(fd), 1e-30)
        assert rel < GEOMETRY_RTOL, (
            f"d tau / d {name}: AD={ad:.10e} vs FD={fd:.10e}, rel={rel:.3e}"
        )


def test_kernel_api_accepts_traced_geometry():
    """The public kernel must be differentiable in the geometry, not only in the
    permittivity."""
    eps = _eps_from_density(_RHO0)

    def objective(params):
        thickness, gap = params[0], params[1]
        return jnp.real(
            kernels.two_body_tau_kernel(
                omega=OMEGA,
                in_plane_wavevector=K_POINT,
                primitive_lattice_vectors=_PLV,
                expansion=_EXPANSION,
                slab_permittivity=eps,
                slab_thickness=thickness,
                gap=gap,
            )
        )

    params = jnp.asarray([THICKNESS, GAP])
    grad = np.asarray(jax.grad(objective)(params))
    assert np.all(np.isfinite(grad)), f"kernel geometry gradient not finite: {grad}"

    for index, step_dir in enumerate(np.eye(2)):
        direction = jnp.asarray(step_dir)
        fd = _directional_fd(objective, params, direction, GEOMETRY_FD_STEP)
        ad = float(np.sum(grad * step_dir))
        rel = abs(ad - fd) / max(abs(fd), 1e-30)
        assert rel < GEOMETRY_RTOL, (
            f"kernel parameter {index}: AD={ad:.10e} vs FD={fd:.10e}, rel={rel:.3e}"
        )


def test_gradients_are_finite_in_degenerate_and_lossless_cases():
    """Uniform layers have massively degenerate eigenvalues, and a lossless
    structure has tau = 0; neither may produce NaN."""

    def uniform(scale):
        eps = (4.0 + 0.5j) * scale * jnp.ones((1, 1), dtype=complex)
        kw = dict(
            wavelength=jnp.asarray(2.0 * jnp.pi / OMEGA),
            in_plane_wavevector=K_POINT[jnp.newaxis, :],
            primitive_lattice_vectors=_PLV,
            expansion=_EXPANSION,
        )
        vac = ss.eigensolve_uniform(**kw, permittivity=1.0 + 0.0j)
        slab = ss.eigensolve_uniform(**kw, permittivity=eps[0, 0])
        R_A, T_A, _ = ss.body_s_matrices(
            vac, slab, jnp.asarray(THICKNESS), is_body_A=True
        )
        R_B, T_B, _ = ss.body_s_matrices(
            vac, slab, jnp.asarray(THICKNESS), is_body_A=False
        )
        F_re, F_ah, F = ht.poynting_flux_matrices(vac)
        P = ht.propagation_matrix(vac.eigenvalues, jnp.asarray(GAP))
        sigma_A = ht.compute_sigma(R_A, T_A, F_re, F_ah)
        sigma_B = ht.compute_sigma(R_B, T_B, F_re, F_ah)
        return jnp.real(jnp.sum(ht.spectral_transfer(sigma_A, sigma_B, P, R_A, R_B, F)))

    point = jnp.asarray(1.0)
    ad = float(jax.grad(uniform)(point))
    fd = float((uniform(point + 1e-6) - uniform(point - 1e-6)) / 2e-6)
    assert np.isfinite(ad), "uniform (degenerate) layer gradient is not finite"
    assert abs(ad - fd) / abs(fd) < GRADIENT_RTOL, (
        f"uniform layer: AD={ad:.10e} vs FD={fd:.10e}"
    )

    def lossless(rho):
        return _tau(rho, formulation=Formulation.FFT, eps_solid=12.0 + 0.0j)

    value = float(lossless(_RHO0))
    grad = np.asarray(jax.grad(lossless)(_RHO0))
    assert abs(value) < 1e-12, f"lossless structure should not transfer: {value:.3e}"
    assert np.all(np.isfinite(grad)), "lossless gradient contains NaN"
    assert np.max(np.abs(grad)) < 1e-10, (
        f"lossless gradient should vanish, got max {np.max(np.abs(grad)):.3e}"
    )


def test_vector_formulation_gradients_are_biased():
    """Known fmmax limitation, pinned so it cannot silently change.

    ``compute_tangent_field`` stop-gradients its input, so for the vector
    formulations ``jax.grad`` is not the derivative of the evaluated objective.
    """
    ratios = []
    for formulation in (Formulation.JONES_FOURIER, Formulation.JONES):

        def objective(rho, formulation=formulation):
            return _tau(rho, formulation=formulation)

        grad = np.asarray(jax.grad(objective)(_RHO0))
        assert np.all(np.isfinite(grad))
        worst = 0.0
        for direction in _unit_directions(4):
            ad = float(np.sum(grad * np.asarray(direction)))
            fd = _directional_fd(objective, _RHO0, direction, FD_STEP)
            ratios.append(ad / fd)
            worst = max(worst, abs(ad - fd) / max(abs(fd), 1e-30))
        assert worst > MIN_JONES_DISAGREEMENT, (
            f"{formulation}: AD now agrees with FD to {worst:.3e}. If fmmax "
            "dropped the stop_gradient on the tangent field, the "
            "Formulation.FFT workaround for optimization is no longer needed."
        )

    assert min(ratios) < 0.0, (
        "expected at least one direction where the stop-gradient gradient has "
        f"the wrong sign; ratios were {ratios}"
    )
