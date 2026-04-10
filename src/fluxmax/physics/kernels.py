"""Two-body RCWA kernels and simple single-device batching helpers.

This module provides a low-level two-body RCWA kernel evaluated at one
``(omega, kx, ky)`` point, plus convenience wrappers for:

- summing over in-plane wavevectors,
- batching over frequencies,
- integrating over frequency with the trapezoidal rule.

The helpers are intentionally simple and target debugging / exploratory runs
on one device. For large production runs (chunking and multi-device sharding),
build a custom driver. See parallelism/ for examples.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype import beartype
from fmmax.basis import Expansion, LatticeVectors  # type: ignore[attr-defined]
from jaxtyping import Array, Complex, Float, jaxtyped

from fluxmax.physics import heat_transfer as ht
from fluxmax.setup import two_body as ss
from fluxmax.units.si_units import omega_nat_to_wavelength_nat



@jaxtyped(typechecker=beartype)
def two_body_tau_kernel(
    omega: Float[Array, ""] | float,
    in_plane_wavevector: Float[Array, "2"],
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
    slab_permittivity: Complex[Array, "ny nx"],
    slab_thickness: float,
    gap: float,
    eps_gap: complex = 1.0 + 0.0j,
) -> Complex[Array, ""]:
    r"""Two-body RCWA transfer kernel for one ``(omega, kx, ky)`` point.

    Geometry is symmetric: both bodies share the same slab permittivity and
    thickness, with a vacuum (or homogeneous) gap medium of permittivity
    ``eps_gap``.

    Parameters
    ----------
    omega : Float[Array, ""] | float
            Angular frequency in natural units.
    in_plane_wavevector : Float[Array, "2"]
            A single in-plane wavevector ``(kx, ky)``.
    primitive_lattice_vectors : LatticeVectors
            Unit-cell lattice vectors for RCWA.
    expansion : Expansion
            Fourier expansion used by fmmax.
    slab_permittivity : Complex[Array, "ny nx"]
            Patterned slab permittivity for this frequency.
    slab_thickness : float
            Slab thickness (same for body A and body B).
    gap : float
            Vacuum gap thickness.
    eps_gap : complex, optional
            Uniform permittivity for the gap-side homogeneous layers.

    Returns
    -------
    Complex[Array, ""]
            Complex transfer value ``tau(omega, kx, ky)``.
    """
    wavelength = omega_nat_to_wavelength_nat(omega)
    thickness = jnp.asarray(slab_thickness)
    gap_thickness = jnp.asarray(gap)

    vac_lsr = ss.eigensolve_uniform(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        permittivity=eps_gap,
    )
    slab_lsr = ss.eigensolve_patterned(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        permittivity_array=slab_permittivity,
    )

    R_A, T_A, _ = ss.body_s_matrices(
        vac_lsr,
        slab_lsr,
        thickness,
        is_body_A=True,
    )
    R_B, T_B, _ = ss.body_s_matrices(
        vac_lsr,
        slab_lsr,
        thickness,
        is_body_A=False,
    )

    F_re, F_ah, F = ht.poynting_flux_matrices(vac_lsr)
    sigma_A = ht.compute_sigma(R_A, T_A, F_re, F_ah)
    sigma_B = ht.compute_sigma(R_B, T_B, F_re, F_ah)
    P = ht.propagation_matrix(vac_lsr.eigenvalues, gap_thickness)
    return ht.spectral_transfer(sigma_A, sigma_B, P, R_A, R_B, F)


def make_two_body_bz_kernel(
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
    slab_thickness: float,
    gap: float,
    eps_gap: complex = 1.0 + 0.0j,
):
    """Build a single-omega BZ kernel closure for the two-body geometry.

    Wraps :func:`two_body_tau_kernel` to build a function of the form
    ``kernel(omega_i, eps_i, k_points)`` that can be passed to
    :func:`fluxmax.parallelism.compute_bz_average` for Brillouin-zone averaging
    over multiple execution modes.

    Parameters
    ----------
    primitive_lattice_vectors : LatticeVectors
        Unit-cell lattice vectors.
    expansion : Expansion
        Fourier expansion.
    slab_thickness : float
        Slab thickness (same for both bodies).
    gap : float
        Vacuum gap thickness.
    eps_gap : complex, optional
        Gap-medium permittivity.

    Returns
    -------
    Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ``kernel(omega_i: scalar, eps_i: (ny, nx), k_points: (n_k, 2))
        -> (n_k,)``
    """

    def kernel(
        omega_i: jnp.ndarray,
        eps_i: jnp.ndarray,
        k_points: jnp.ndarray,
    ) -> jnp.ndarray:
        tau_k = two_body_tau_per_k(
            omega=omega_i,
            in_plane_wavevectors=k_points,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            slab_permittivity=eps_i,
            slab_thickness=slab_thickness,
            gap=gap,
            eps_gap=eps_gap,
        )
        return jnp.real(tau_k)

    return kernel


@jaxtyped(typechecker=beartype)
def two_body_tau_per_k(
    omega: Float[Array, ""] | float,
    in_plane_wavevectors: Float[Array, "*kbatch 2"],
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
    slab_permittivity: Complex[Array, "ny nx"],
    slab_thickness: float,
    gap: float,
    eps_gap: complex = 1.0 + 0.0j,
) -> Complex[Array, " n_k"]:
    """Evaluate :func:`two_body_tau_kernel` over a batch of k-points."""
    k_points = jnp.reshape(in_plane_wavevectors, (-1, 2))
    return jax.vmap(
        lambda kvec: two_body_tau_kernel(
            omega=omega,
            in_plane_wavevector=kvec,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            slab_permittivity=slab_permittivity,
            slab_thickness=slab_thickness,
            gap=gap,
            eps_gap=eps_gap,
        )
    )(k_points)


@jaxtyped(typechecker=beartype)
def two_body_k_integrated_tau(
    omega: Float[Array, ""] | float,
    in_plane_wavevectors: Float[Array, "*kbatch 2"],
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
    slab_permittivity: Complex[Array, "ny nx"],
    slab_thickness: float,
    gap: float,
    eps_gap: complex = 1.0 + 0.0j,
    average: bool = True,
) -> Float[Array, ""]:
    """Sum or average ``Re[tau]`` over in-plane wavevectors."""
    tau_k = two_body_tau_per_k(
        omega=omega,
        in_plane_wavevectors=in_plane_wavevectors,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        slab_permittivity=slab_permittivity,
        slab_thickness=slab_thickness,
        gap=gap,
        eps_gap=eps_gap,
    )
    tau_real_sum = jnp.sum(jnp.real(tau_k))
    if average:
        return tau_real_sum / tau_k.size
    return tau_real_sum


def broadcast_slab_permittivity(
    slab_permittivity: jnp.ndarray,
    n_omega: int,
) -> jnp.ndarray:
    """Normalize slab permittivity to shape ``(n_omega, ny, nx)``."""
    eps = jnp.asarray(slab_permittivity, dtype=complex)
    if eps.ndim == 2:
        return jnp.broadcast_to(eps[jnp.newaxis, ...], (n_omega, *eps.shape))
    if eps.ndim == 3 and eps.shape[0] == n_omega:
        return eps
    raise ValueError(
        "Expected slab_permittivity with shape (ny, nx) or (n_omega, ny, nx), "
        f"got {eps.shape} for n_omega={n_omega}."
    )


# Keep the old private name as an alias for backwards compatibility.
_broadcast_slab_permittivity_over_omega = broadcast_slab_permittivity


@jaxtyped(typechecker=beartype)
def two_body_omega_batched_tau(
    omega: Float[Array, " n_omega"],
    in_plane_wavevectors: Float[Array, "*kbatch 2"],
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
    slab_permittivity: Complex[Array, "..."],
    slab_thickness: float,
    gap: float,
    eps_gap: complex = 1.0 + 0.0j,
    average_k: bool = True,
) -> Float[Array, " n_omega"]:
    """Batch :func:`two_body_k_integrated_tau` over frequency points."""
    omega = jnp.asarray(omega)
    eps_omega = broadcast_slab_permittivity(slab_permittivity, omega.size)

    return jax.vmap(
        lambda omega_i, eps_i: two_body_k_integrated_tau(
            omega=omega_i,
            in_plane_wavevectors=in_plane_wavevectors,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            slab_permittivity=eps_i,
            slab_thickness=slab_thickness,
            gap=gap,
            eps_gap=eps_gap,
            average=average_k,
        )
    )(omega, eps_omega)


@jaxtyped(typechecker=beartype)
def frequency_integrated_two_body_tau(
    omega: Float[Array, " n_omega"],
    in_plane_wavevectors: Float[Array, "*kbatch 2"],
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
    slab_permittivity: Complex[Array, "..."],
    slab_thickness: float,
    gap: float,
    eps_gap: complex = 1.0 + 0.0j,
    average_k: bool = True,
) -> Float[Array, ""]:
    """Integrate the omega-batched two-body transfer with trapezoidal rule."""
    tau_omega = two_body_omega_batched_tau(
        omega=omega,
        in_plane_wavevectors=in_plane_wavevectors,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        slab_permittivity=slab_permittivity,
        slab_thickness=slab_thickness,
        gap=gap,
        eps_gap=eps_gap,
        average_k=average_k,
    )
    return jnp.trapezoid(tau_omega, x=omega)




__all__ = [
    "omega_to_wavelength",
    "two_body_tau_kernel",
    "two_body_tau_per_k",
    "two_body_k_integrated_tau",
    "broadcast_slab_permittivity",
    "two_body_omega_batched_tau",
    "frequency_integrated_two_body_tau",
    "make_two_body_bz_kernel",
]
