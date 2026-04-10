"""
Helpers: build fmmax layer stacks for two-body NFRHT geometry.

Geometry (z increases left → right):

   |  above A  |   slab A   |  vacuum gap  |   slab B   |  above B  |
   | (semi-∞)  | thickness  |      d       | thickness  | (semi-∞)  |
   z=-∞                    z=0            z=d          z=d+d_B     z=+∞

S-matrix element mapping (vacuum bounding layers have thickness 0,
so S-matrix blocks are in the plane-wave basis directly):

  Body A  stack = [vac_above, slab_A, vac_gap]
    R_A  = s_matrix_A.s12   (reflection back into gap)
    T_A  = s_matrix_A.s22   (transmission through A, gap→above)

  Body B  stack = [vac_gap, slab_B, vac_above]
    R_B  = s_matrix_B.s21   (reflection back into gap)
    T_B  = s_matrix_B.s11   (transmission through B, gap→above)
"""

from __future__ import annotations

import jax.numpy as jnp

# We meed to do this import because fmmax doesn't expose these functions in __all__.
from fmmax._fmm_result import LayerSolveResult
from fmmax.basis import (  # type: ignore[attr-defined]
    Expansion,
    LatticeVectors,
    Truncation,
    X,
    Y,
    brillouin_zone_in_plane_wavevector,
    generate_expansion,
)
from fmmax.fmm import (  # type: ignore[attr-defined]
    Formulation,
    eigensolve_isotropic_media,
)
from fmmax.scattering import (  # type: ignore[attr-defined]
    ScatteringMatrix,
    stack_s_matrix,
)


def make_rcwa_setup(
    pitch: float,
    approximate_num_terms: int,
    brillouin_grid_shape: tuple[int, int] = (1, 1),
) -> tuple[LatticeVectors, Expansion, jnp.ndarray]:
    """Create lattice vectors, expansion, and in-plane wavevectors.

    Parameters
    ----------
    pitch : float
        The side length of the square unit cell.
    approximate_num_terms : int
        The approximate number of Fourier terms to include in the expansion.
        This controls the accuracy of the RCWA calculation.
        More terms means better accuracy but higher computational cost.
    brillouin_grid_shape : tuple[int, int], optional
        The shape of the grid used to sample the Brillouin zone (BZ) in the
        in-plane wavevector space. The default is (1, 1), which corresponds to sampling
        only the Gamma point.

    Returns
    -------
    primitive_lattice_vectors, expansion, in_plane_wavevector

    Notes
    -----
    1. Define primitive lattice vectors for a square lattice in the XY plane.
       These will be used by fmmax to construct the reciprocal lattice.
    2. Generate the Fourier expansion for the RCWA method using fmmax's
       `generate_expansion` function. This determines how many Fourier components
       (plane waves) are included in the calculation, based on the specified
       `approximate_num_terms` and the geometry of the lattice. Circular means we
       include all Fourier components within a circle in reciprocal space, which is a
       common choice for isotropic structures.
    3. Compute the in-plane wavevectors that sample the Brillouin zone (BZ) using
       fmmax's `brillouin_zone_in_plane_wavevector` function. This creates a grid of
       k-points in the reciprocal space corresponding to the specified
       `brillouin_grid_shape`.
    """
    primitive_lattice_vectors = LatticeVectors(u=pitch * X, v=pitch * Y)
    expansion = generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_num_terms,
        truncation=Truncation.CIRCULAR,
    )
    in_plane_wavevector = brillouin_zone_in_plane_wavevector(
        brillouin_grid_shape, primitive_lattice_vectors
    )
    return primitive_lattice_vectors, expansion, in_plane_wavevector


def eigensolve_uniform(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
    permittivity: complex,
    formulation: Formulation = Formulation.FFT,
) -> LayerSolveResult:
    """Eigensolve a uniform (homogeneous) layer.

    For a uniform layer, the permittivity is constant across the unit cell.
    Internally we call :func:`fmmax.fmm.eigensolve_isotropic_media` with a
    permittivity array of shape ``(1, 1)``.

    Parameters
    ----------
    wavelength : jnp.ndarray
        Free-space wavelength(s).
    in_plane_wavevector : jnp.ndarray
        In-plane wavevector(s) ``(kx0, ky0)``.
    primitive_lattice_vectors : LatticeVectors
        Real-space lattice vectors for the unit cell.
    expansion : Expansion
        Fourier expansion specifying the set of plane-wave orders.
    permittivity : complex
        Uniform relative permittivity for the layer.
    formulation : Formulation, optional
        RCWA formulation used by fmmax. Default is ``Formulation.FFT``.

    Returns
    -------
    LayerSolveResult
        Eigensolve result containing eigenvalues/eigenvectors for the layer.

    Notes
    -----
    The primitive lattice vectors are larger than the permittivity array.
    The lattice vectors encode the unit-cell size in continuous space and are
    separate from the discretized permittivity sampling.

    The permittivity array is a 2-D grid assumed to span exactly one real-space
    unit cell defined by the primitive lattice vectors. For a uniform layer we
    only need a single permittivity value, so we create a ``(1, 1)`` array.

    If your resolution is not big enough to capture
    ``min_array_shape_for_expansion`` inside fmmax (i.e., the minimum size of
    the permittivity array needed to accurately represent the Fourier
    expansion), you'll get a warning or error anyway.

    This is all fmmax needs: it will then compute the Fourier expansion based
    on this grid.
    """
    return eigensolve_isotropic_media(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivity=jnp.asarray(permittivity, dtype=complex)[jnp.newaxis, jnp.newaxis],
        expansion=expansion,
        formulation=formulation,
    )


def eigensolve_patterned(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
    permittivity_array: jnp.ndarray,
    formulation: Formulation = Formulation.JONES_FOURIER,
) -> LayerSolveResult:
    """Eigensolve a patterned layer from a 2-D permittivity array.

    Same as above but now the permittivity is given by a 2-D array representing the
    spatial distribution of permittivity within the unit cell.
    """
    return eigensolve_isotropic_media(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivity=permittivity_array,
        expansion=expansion,
        formulation=formulation,
    )


def body_s_matrices(
    vac_lsr: LayerSolveResult,
    slab_lsr: LayerSolveResult,
    slab_thickness: jnp.ndarray,
    is_body_A: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, ScatteringMatrix]:
    """Compute S-matrix for one body and extract R and T.

    For body A the stack is [vac_above, slab_A, vac_gap]:
        R_A = s.s12,  T_A = s.s22
    For body B the stack is [vac_gap, slab_B, vac_above]:
        R_B = s.s21,  T_B = s.s11

    We are passing zero as the thicknesses because fmmax interprets this as not
    propagating fields in this segement,
    just to evaluate the S-matrix at those interfaces.

    slab_thickness may be array to support vectorized computation.
    """
    zero = jnp.zeros_like(slab_thickness)

    if is_body_A:
        s = stack_s_matrix(
            layer_solve_results=[vac_lsr, slab_lsr, vac_lsr],
            layer_thicknesses=[zero, slab_thickness, zero],
        )
        R = s.s12  # reflection back into gap
        T = s.s22  # transmission gap -> above A
    else:
        s = stack_s_matrix(
            layer_solve_results=[vac_lsr, slab_lsr, vac_lsr],
            layer_thicknesses=[zero, slab_thickness, zero],
        )
        R = s.s21  # reflection back into gap
        T = s.s11  # transmission gap -> above B
    return R, T, s


def cell_area(primitive_lattice_vectors: LatticeVectors) -> jnp.ndarray:
    """Area of the real-space unit cell |u x v|.

    Necessary for some tests and fmmax doesn't expose a public function for this.

    The ... are just for JAX batching purposes, so we can compute the area for multiple
    sets of lattice vectors at once if needed.
    """
    u = primitive_lattice_vectors.u
    v = primitive_lattice_vectors.v
    return jnp.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
