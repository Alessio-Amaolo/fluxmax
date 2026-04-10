"""Core heat transfer trace formula via RCWA.

Implements the spectral heat flux between two finite-thickness periodic
bodies separated by a vacuum gap.

The S-matrices from fmmax operate on H-field Fourier amplitudes ``{Hx, Hy}``
per diffraction order. In this basis the Poynting-flux bilinear form is a
full, non-diagonal matrix ``F = A† φ`` rather than simply ``diag(kz)``. The
trace formula is

    ``T = Tr[P† D† Σ_A D P (F†)^-1 Σ_B F^-1]``

Units
-----
This module uses the natural-unit convention:

- ``c = 1``, ``hbar = 1``, ``k_B = 1``
- All lengths are expressed in an arbitrary base unit ``L0``.
"""

import jax.numpy as jnp
from beartype import beartype
from fmmax._fmm_result import LayerSolveResult
from jaxtyping import Array, Complex, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def poynting_flux_matrices(
    vac_lsr: LayerSolveResult,
) -> tuple[
    Complex[Array, "*batch two_n two_n"],
    Complex[Array, "*batch two_n two_n"],
    Complex[Array, "*batch two_n two_n"],
]:
    r"""Poynting-flux bilinear-form matrices for the vacuum gap.

    The time-averaged z-directed Poynting flux for forward ($a$) and backward
    ($b$) modal amplitudes co-located at the same z-plane is

    .. math::
        S_z = \tfrac12 \operatorname{Re}\!\bigl[(a-b)^\dagger \mathcal F\,(a+b)\bigr],

    where $\mathcal F = A^\dagger \phi$ with $A$ the Poynting "A-matrix" from
    [2012 Liu] and $\phi$ the eigenvector matrix of the vacuum layer.

    Parameters
    ----------
    vac_lsr : LayerSolveResult
        Layer-solve result for the **vacuum** gap layer.

    Returns
    -------
    F_re : Complex[Array, "*batch two_n two_n"]
        Hermitian part of the flux matrix, shape ``(..., 2N, 2N)``.
    F_ah : Complex[Array, "*batch two_n two_n"]
        Anti-Hermitian part divided by *i*, shape ``(..., 2N, 2N)``.  Real and
        Hermitian; non-zero only for evanescent modes.
    F : Complex[Array, "*batch two_n two_n"]
        Full (complex, generally non-Hermitian) flux matrix
        $\mathcal F = A^\dagger \phi$, shape ``(..., 2N, 2N)``.  Needed by
        :func:`spectral_transfer` for the noise-correlator normalization.
    """
    q = vac_lsr.eigenvalues
    phi = vac_lsr.eigenvectors
    omega_k = vac_lsr.omega_script_k_matrix
    omega = wavelength_to_omega(vac_lsr.wavelength)

    inv_wq = 1.0 / (omega[..., jnp.newaxis] * q)
    A = omega_k @ (phi * inv_wq[..., jnp.newaxis, :])
    F = _adjoint(A) @ phi

    F_re = (F + _adjoint(F)) / 2
    F_ah = (F - _adjoint(F)) / (2j)
    return F_re, F_ah, F


@jaxtyped(typechecker=beartype)
def compute_sigma(
    R: Complex[Array, "*batch two_n two_n"],
    T: Complex[Array, "*batch two_n two_n"],
    F_re: Complex[Array, "*batch two_n two_n"],
    F_ah: Complex[Array, "*batch two_n two_n"],
) -> Complex[Array, "*batch two_n two_n"]:
    r"""Emission operator for a finite-thickness body.

    Uses the Poynting-flux bilinear form in fmmax's H-field modal basis.

    .. math::
        \Sigma = \mathcal F_{Re}
                 - R^\dagger \mathcal F_{Re}\, R
                 + i\bigl(\mathcal F_{AH}\, R - R^\dagger \mathcal F_{AH}\bigr)
                 - T^\dagger \mathcal F_{Re}\, T

    This is guaranteed PSD for any passive scatterer because
    $\frac12 b^\dagger \Sigma\, b$ equals the absorbed power for incident
    amplitudes $b$.

    For a semi-infinite body, pass ``T = 0``.

    Parameters
    ----------
    R : Complex[Array, "*batch two_n two_n"]
        Reflection matrix of the body as seen from the gap, ``(..., 2N, 2N)``.
    T : Complex[Array, "*batch two_n two_n"]
        Transmission matrix through the body, ``(..., 2N, 2N)``.
    F_re : Complex[Array, "*batch two_n two_n"]
        Hermitian part of the Poynting flux matrix, ``(..., 2N, 2N)``.
    F_ah : Complex[Array, "*batch two_n two_n"]
        Anti-Hermitian part / *i* of the Poynting flux matrix, ``(..., 2N, 2N)``.

    Returns
    -------
    Σ : Complex[Array, "*batch two_n two_n"]
        Hermitian emission operator, shape ``(..., 2N, 2N)``.
    """
    R_dag = _adjoint(R)
    T_dag = _adjoint(T)

    sigma = F_re - R_dag @ F_re @ R - 1j * (F_ah @ R - R_dag @ F_ah) - T_dag @ F_re @ T
    return sigma


@jaxtyped(typechecker=beartype)
def propagation_matrix(
    eigenvalues: Complex[Array, "*batch two_n"],
    gap_thickness: Float[Array, "..."] | float,
) -> Complex[Array, "*batch two_n two_n"]:
    """Diagonal propagation matrix  P = diag(exp(i q d)).

    For propagating modes (Im q = 0, Re q > 0):  oscillatory phase.
    For evanescent modes (Im q > 0):             exponential decay.

    Can be batched over multiple sets of eigenvalues and gap thicknesses, as
    long as the last axis of ``eigenvalues`` matches the size of the square
    ``P`` matrix.

    Parameters
    ----------
    eigenvalues : Complex[Array, "*batch two_n"]
        Vacuum-gap kz eigenvalues with shape (..., 2N).
    gap_thickness : Float[Array, "*batch"] | float
        Gap thickness with shape (...,). Broadcastable to eigenvalues.

    Returns
    -------
    Complex[Array, "*batch two_n two_n"]
        Diagonal propagation matrix with shape (..., 2N, 2N).
    """
    gap_thickness_array: Float[Array, "..."] = jnp.asarray(gap_thickness)
    return _diag(jnp.exp(1j * eigenvalues * gap_thickness_array[..., jnp.newaxis]))


@jaxtyped(typechecker=beartype)
def spectral_transfer(
    sigma_A: Complex[Array, "*batch two_n two_n"],
    sigma_B: Complex[Array, "*batch two_n two_n"],
    P: Complex[Array, "*batch two_n two_n"],
    R_A: Complex[Array, "*batch two_n two_n"],
    R_B: Complex[Array, "*batch two_n two_n"],
    F: Complex[Array, "*batch two_n two_n"],
) -> Complex[Array, "*batch"]:
    r"""
    Compute the spectral transmission factor.

        Tr[ P† D† Σ_A D P (F†)⁻¹ Σ_B F⁻¹ ],

    where

        D = (I - P R_B P R_A)^{-1},

    and F = A† φ is the full Poynting-flux matrix from
    :func:`poynting_flux_matrices`.

    In the TE/TM plane-wave basis F is diagonal with entries ω/kz (TE) and
    kz/ω (TM), and the formula reduces to the standard Polder-Van Hove
    expression with |K|⁻¹ = diag(1/|kz|).

    The physical spectral heat flux is

        Φ(ω)/A = hbar omega Θ(ω, T) * (1 / (N_BZ A_cell))
                 * Σ_k Re[spectral_transfer(k)].

    where k is a sum over the Brillouin zone and N_BZ is the number of
    k points in the BZ grid.

    Parameters
    ----------
    sigma_A : Complex[Array, "*batch two_n two_n"]
        Emission operator for body A with shape (..., 2N, 2N).
    sigma_B : Complex[Array, "*batch two_n two_n"]
        Emission operator for body B with shape (..., 2N, 2N).
    P : Complex[Array, "*batch two_n two_n"]
        Gap propagation matrix with shape (..., 2N, 2N).
    R_A : Complex[Array, "*batch two_n two_n"]
        Reflection matrix of body A with shape (..., 2N, 2N).
    R_B : Complex[Array, "*batch two_n two_n"]
        Reflection matrix of body B with shape (..., 2N, 2N).
    F : Complex[Array, "*batch two_n two_n"]
        Full Poynting-flux matrix F = A† φ with shape (..., 2N, 2N),
        as returned by :func:`poynting_flux_matrices`.

    Returns
    -------
    Complex[Array, "*batch"]
        Complex scalar per batch element with shape (...,). The physical
        contribution is obtained by taking the real part and summing
        over the Brillouin zone.
    """
    n = P.shape[-1]
    Id = jnp.eye(n, dtype=P.dtype)
    M = Id - P @ R_B @ P @ R_A
    D = jnp.linalg.solve(M, Id)
    P_dag = _adjoint(P)
    D_dag = _adjoint(D)

    # Noise correlator: (F†)⁻¹ Σ_B F⁻¹
    F_inv = jnp.linalg.solve(F, Id)
    sigma_B_tilde = _adjoint(F_inv) @ sigma_B @ F_inv

    W = P_dag @ D_dag @ sigma_A @ D @ P @ sigma_B_tilde
    return _trace(W)


@jaxtyped(typechecker=beartype)
def bose_einstein(
    omega_nat: Float[Array, "*shape"] | float,
    T_nat: Float[Array, "*shape"] | float,
) -> Float[Array, "*shape"]:
    r"""Bose-Einstein mean occupation in natural units.

    In the convention hbar = k_B = 1 this is

    $$\Theta(\omega, T) = \frac{1}{\exp(\omega / T) - 1}.$$

    Parameters
    ----------
    omega_nat
        Angular frequency in units of 1/L0.
    T_nat
        Temperature in the same units as ``omega_nat``.

    Returns
    -------
    Float[Array, "*shape"]
        Mean occupation number (dimensionless), broadcast over inputs.
    """
    x = omega_nat / T_nat
    return 1.0 / jnp.expm1(x)


@jaxtyped(typechecker=beartype)
def spectral_heat_flux(
    normalized_transfer: Float[Array, "*shape"],
    omega_nat: Float[Array, "*shape"] | float,
    T_nat: Float[Array, "*shape"] | float,
    cell_area: Float[Array, ""] | float,
    n_bz: int,
) -> Float[Array, "*shape"]:
    r"""Compute the natural-unit spectral heat flux per unit area.

    In natural units ($\hbar = k_B = 1$), the spectral flux per area is

    $$
    \Phi(\omega)/A = \omega\Theta(\omega, T)
    \frac{1}{N_{\mathrm{BZ}} A_{\mathrm{cell}}}
    \sum_k \tau(\omega, k).
    $$

    Parameters
    ----------
    normalized_transfer : Float[Array, "*shape"]
        Transfer values already summed over Brillouin-zone points, or an
        array of per-$k$ values. This function applies the BZ average
        ``(1 / n_bz) * sum_k``.
    omega_nat : Float[Array, "*shape"] | float
        Angular frequency in units of ``1/L0``.
    T_nat : Float[Array, "*shape"] | float
        Temperature in the same units as ``omega_nat``.
    cell_area : Float[Array, ""] | float
        Unit cell area in units of ``L0**2``.
    n_bz : int
        Number of Brillouin-zone sample points (e.g. ``Nu * Nv``).

    Returns
    -------
    flux : Float[Array, "*shape"]
        Spectral heat flux per unit area in natural units.

    Notes
    -----
    Conversion to SI requires an external scaling that depends on the
    choice of length unit ``L0``; see ``si_units.py``.
    """
    theta = bose_einstein(omega_nat, T_nat)
    prefactor = omega_nat * theta
    bz_avg = jnp.sum(normalized_transfer) / n_bz
    return prefactor * bz_avg / cell_area


@jaxtyped(typechecker=beartype)
def wavelength_to_omega(
    wavelength: Float[Array, "*shape"] | float,
) -> Float[Array, "*shape"]:
    r"""Convert wavelength to angular frequency in fmmax natural units.

    Uses the convention ``c = 1`` so that $\omega = 2\pi / \lambda$.

    Parameters
    ----------
    wavelength : Float[Array, "*shape"] | float
        Wavelength in units of ``L0``.

    Returns
    -------
    omega_nat : Float[Array, "*shape"]
        Angular frequency in units of ``1/L0``.
    """
    return jnp.asarray(2.0 * jnp.pi / wavelength)


@jaxtyped(typechecker=beartype)
def _diag(x: Complex[Array, "*batch n"]) -> Complex[Array, "*batch n n"]:
    """Create a batched diagonal matrix from the last axis.

    Parameters
    ----------
    x : Complex[Array, "*batch n"]
        Input array with shape ``(..., n)``.

    Returns
    -------
    y : Complex[Array, "*batch n n"]
        Array with shape ``(..., n, n)`` whose last two axes form a
        diagonal matrix with diagonal entries taken from ``x``.
    """
    shape = x.shape + (x.shape[-1],)
    y = jnp.zeros(shape, x.dtype)
    i = jnp.arange(x.shape[-1])
    return y.at[..., i, i].set(x)


@jaxtyped(typechecker=beartype)
def _adjoint(
    x: Complex[Array, "*batch m n"],
) -> Complex[Array, "*batch n m"]:
    """Compute the conjugate transpose over the last two axes.

    Parameters
    ----------
    x : Complex[Array, "*batch m n"]
        Input array with shape ``(..., m, n)``.

    Returns
    -------
    x_dag : Complex[Array, "*batch n m"]
        Conjugate-transposed array with shape ``(..., n, m)``.
    """
    return jnp.conj(jnp.swapaxes(x, -2, -1))


@jaxtyped(typechecker=beartype)
def _trace(x: Complex[Array, "*batch n n"]) -> Complex[Array, "*batch"]:
    """Compute the trace over the last two axes.

    Parameters
    ----------
    x : Complex[Array, "*batch n n"]
        Input array with shape ``(..., n, n)``.

    Returns
    -------
    tr : Complex[Array, "*batch"]
        Trace with shape ``(...,)``.
    """
    return jnp.trace(x, axis1=-2, axis2=-1)
