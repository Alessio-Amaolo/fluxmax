"""Design tools and parameterizations for topological optimization."""

from typing import Union

import jax.numpy as jnp
from beartype import beartype
from fmmax.utils import interpolate_permittivity  # type: ignore[attr-defined]
from jaxtyping import Array, Complex, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def project_tanh(
    rho_tilde: Float[Array, "..."],
    beta: Union[Float[Array, ""], float],
    eta: float = 0.5,
) -> Float[Array, "..."]:
    """
    Projects a continuous density field using a tanh filter.

    Parameters
    ----------
    rho_tilde : Float[Array, "..."]
        The unprojected density field.
    beta : Float[Array, ""] or float
        The projection steepness parameter.
    eta : float, optional
        The projection threshold, by default 0.5.

    Returns
    -------
    Float[Array, "..."]
        The projected density field.
    """
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (rho_tilde - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den


@jaxtyped(typechecker=beartype)
def metallic_eps_from_density(
    rho: Float[Array, "..."],
    eps_solid: Union[complex, Complex[Array, "..."]],
    eps_void: Union[complex, Complex[Array, "..."]],
) -> Complex[Array, "..."]:
    """
    Calculate the effective permittivity of a metallic topology.

    Parameters
    ----------
    rho : Float[Array, "..."]
        The projected density field, with values in [0, 1].
    eps_solid : complex or Complex[Array, "..."]
        The permittivity of the solid material.
    eps_void : complex or Complex[Array, "..."]
        The permittivity of the void material.

    Returns
    -------
    Complex[Array, "..."]
        The effective permittivity array.
    """
    rho = jnp.clip(rho, 0.0, 1.0)
    eps_void_arr = jnp.asarray(eps_void, dtype=complex)

    if jnp.ndim(eps_solid) == 0:
        eps_solid_arr = jnp.asarray(eps_solid, dtype=complex)
        return interpolate_permittivity(
            permittivity_solid=eps_solid_arr,
            permittivity_void=eps_void_arr,
            density=rho,
        )

    eps_solid_arr = jnp.asarray(eps_solid, dtype=complex)
    rho_batched = jnp.broadcast_to(rho, eps_solid_arr.shape + rho.shape)
    return interpolate_permittivity(
        permittivity_solid=eps_solid_arr[..., jnp.newaxis, jnp.newaxis],
        permittivity_void=eps_void_arr,
        density=rho_batched,
    )


@jaxtyped(typechecker=beartype)
def dielectric_eps_from_density(
    rho: Float[Array, "..."],
    eps_solid: Union[complex, Complex[Array, "..."]],
    eps_void: Union[complex, Complex[Array, "..."]],
) -> Complex[Array, "..."]:
    """Linearly interpolate permittivity between void and solid values.

    The density is clipped to ``[0, 1]`` so that ``rho = 0`` returns
    ``eps_void`` and ``rho = 1`` returns ``eps_solid``.
    """
    rho = jnp.clip(rho, 0.0, 1.0)
    eps_solid_arr = jnp.asarray(eps_solid, dtype=complex)
    eps_void_arr = jnp.asarray(eps_void, dtype=complex)
    return eps_void_arr + rho * (eps_solid_arr - eps_void_arr)


def circular_inclusion_permittivity(
    pitch: float,
    diameter: float,
    eps_host: complex,
    eps_inclusion: complex,
    resolution: float,
    softness: float = 0.0,
) -> jnp.ndarray:
    r"""
    2-D permittivity array with a circular inclusion centered in a square unit cell.

    Creates a discretized grid representing the relative permittivity
    of a square domain. Pixels within the circle's radius are assigned
    `eps_inclusion`, while the remaining pixels are assigned `eps_host`.

    Parameters
    ----------
    pitch : float
        The side length of the square unit cell (e.g., in micrometers).
    diameter : float
        The diameter of the circular inclusion centered at (0, 0).
    eps_host : complex
        The relative permittivity of the background (host) medium.
    eps_inclusion : complex
        The relative permittivity of the circular inclusion material.
    resolution : float
        The physical size of a single pixel/grid cell.
    softness : float, optional
        Width of the softened boundary region in the same units as ``pitch``
        and ``diameter``. The default of ``0.0`` preserves the current hard
        circular boundary.

    Returns
    -------
    jnp.ndarray
        A 2-D JAX array of shape (ny, nx) containing the permittivity values,
        where nx = ny = round(pitch / resolution).

    See Also
    --------
    eigensolve_isotropic_media : Function that typically consumes this array.
    """
    if softness < 0.0:
        raise ValueError("softness must be non-negative")

    nx = int(round(pitch / resolution))
    ny = int(round(pitch / resolution))
    x = (jnp.arange(nx) + 0.5) * resolution - pitch / 2.0
    y = (jnp.arange(ny) + 0.5) * resolution - pitch / 2.0
    xx, yy = jnp.meshgrid(x, y, indexing="xy")
    radius = diameter / 2.0

    if softness == 0.0:
        mask = (xx**2 + yy**2) < radius**2
        return jnp.where(mask, eps_inclusion, eps_host)

    signed_distance = radius - jnp.sqrt(xx**2 + yy**2)
    density = jnp.clip(0.5 + signed_distance / softness, 0.0, 1.0)
    return dielectric_eps_from_density(
        rho=density,
        eps_solid=eps_inclusion,
        eps_void=eps_host,
    )


def circular_exclusion_permittivity(
    pitch: float,
    diameter: float,
    eps_slab: complex,
    eps_exclusion: complex,
    resolution: float,
    softness: float = 0.0,
) -> jnp.ndarray:
    r"""
    2-D permittivity array with a circular exclusion centered in a slab unit cell.

    This is the hole/exclusion counterpart to
    :func:`circular_inclusion_permittivity`. Pixels inside the circular
    exclusion are assigned ``eps_exclusion`` while the surrounding slab is
    assigned ``eps_slab``.

    Parameters
    ----------
    pitch : float
        The side length of the square unit cell.
    diameter : float
        The diameter of the circular exclusion centered at ``(0, 0)``.
    eps_slab : complex
        The relative permittivity of the slab material.
    eps_exclusion : complex
        The relative permittivity inside the excluded region.
    resolution : float
        The physical size of a single pixel/grid cell.
    softness : float, optional
        Width of the softened boundary region in the same units as ``pitch``
        and ``diameter``. The default of ``0.0`` preserves the current hard
        circular boundary.

    Returns
    -------
    jnp.ndarray
        A 2-D JAX array of shape ``(ny, nx)`` containing the permittivity values.
    """
    return circular_inclusion_permittivity(
        pitch=pitch,
        diameter=diameter,
        eps_host=eps_slab,
        eps_inclusion=eps_exclusion,
        resolution=resolution,
        softness=softness,
    )


