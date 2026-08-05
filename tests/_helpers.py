"""Setup helpers shared by more than one test module.

Not a test module itself.

* :func:`stack_blocks`: R and T of a *multi-layer* body, using the same
  S-matrix block convention as :func:`~fluxmax.setup.two_body.body_s_matrices`.
  Needed wherever a body is not a single layer, i.e. wherever the body is not
  mirror-symmetric in z and the convention actually has consequences.
* :func:`centered_grating`: the 1D stripe permittivity profile.
* :data:`K_GENERIC`: the off-axis in-plane wavevector used for mode-mixing
  tests.
"""

import jax.numpy as jnp
import numpy as np
from fmmax.fields import directional_poynting_flux  # type: ignore[attr-defined]
from fmmax.scattering import stack_s_matrix

NX = 64

# Generic off-axis in-plane wavevector. Off the k_x axis on purpose: at
# k_par = (k, 0) fmmax's (Hx, Hy) channels line up with TE/TM, so every 2x2 block
# of the flux form F, of Sigma and of the correlator is diagonal and any error in
# the polarization-mixing part is invisible. Also clear of the light line for the
# pitches used here (nearest channel sits at |k + G|/omega ~ 0.96 at pitch = 1).
K_GENERIC = np.array([0.37 * 2 * np.pi, 0.11 * 2 * np.pi])


def net_flux(vac, forward, backward) -> float:
    """Net z-directed Poynting flux of a mode-amplitude pair, from fmmax.

    The reference used wherever a test needs absorbed power without going through
    :mod:`fluxmax.physics.heat_transfer`.
    """
    fwd, bwd = directional_poynting_flux(
        jnp.asarray(forward)[:, jnp.newaxis], jnp.asarray(backward)[:, jnp.newaxis], vac
    )
    return float(jnp.real(jnp.sum(fwd) + jnp.sum(bwd)))


def stack_blocks(vac, layers, thicknesses, *, is_body_A):
    """``(R_gap_side, T_gap_side, T_far_side)`` of a multi-layer body.

    Same block convention as :func:`~fluxmax.setup.two_body.body_s_matrices`
    (``R_A = s12, T_A = s22`` for the body left of the gap; ``R_B = s21,
    T_B = s11`` for the body right of it), extended to a stack. The extra
    far-side transmission is returned so tests can assert that the two
    directions really do differ -- otherwise a swapped block is undetectable.

    ``layers`` is ordered along +z, so for body A (gap on its right) that is
    far -> near, and for body B (gap on its left) it is near -> far.
    """
    zero = jnp.zeros(())
    solves = [vac, *layers, vac]
    thick = [zero, *thicknesses, zero]
    s = stack_s_matrix(layer_solve_results=solves, layer_thicknesses=thick)
    if is_body_A:
        return s.s12, s.s22, s.s11
    return s.s21, s.s11, s.s22


def centered_grating(
    eps_stripe: complex, fill: float, *, center: float = 0.5, nx: int = NX
):
    """1-D stripe grating of duty cycle ``fill``, centered on ``center``.

    Returns an ``(nx, nx)`` permittivity array: constant along y, so the
    structure is a grating rather than a 2-D pattern.
    """
    x = (np.arange(nx) + 0.5) / nx
    start = center - fill / 2
    profile = np.where(((x - start) % 1.0) < fill, eps_stripe, 1.0 + 0.0j)
    return jnp.asarray(np.tile(profile[:, None], (1, nx)), dtype=complex)
