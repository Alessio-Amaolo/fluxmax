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
