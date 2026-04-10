"""Analytical Lifshitz formula for radiative heat transfer.

Planar bodies separated by a vacuum gap.

Supports both finite-thickness slabs and semi-infinite half-spaces.
Provides the spectral transfer function T(ω, kpar) and its integral
over kpar for the unpatterned (planar) case.

See Polder & Van Hove, Theory of Radiative Heat Transfer between Closely Spaced Bodies.
"""

from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

Polarization = Literal["s", "p"]


@jaxtyped(typechecker=beartype)
def _kz(
    eps: Complex[Array, "..."] | complex,
    omega: Float[Array, "..."] | float,
    kpar: Float[Array, "..."] | float,
) -> Complex[Array, "..."]:
    r"""Compute the normal wavevector component $k_z$.

    Uses

    $$k_z = \sqrt{\varepsilon\,\omega^2 - k_{\parallel}^2},$$

    and enforces the physical branch condition $\Im(k_z) \ge 0$ (decaying
    evanescent waves).

    Parameters
    ----------
    eps : complex or array
        Relative permittivity $\varepsilon$.
        Accepts a Python ``complex`` or a ``jaxtyping`` complex array
        (``Complex[Array, "*shape"]``). Must be broadcastable with ``omega`` and
        ``kpar``.
    omega : float or array
        Angular frequency $\omega$ (same broadcast rules as above).
    kpar : float or array
        In-plane wavevector magnitude $k_{\parallel}$.

    Returns
    -------
    kz : complex array
        Normal wavevector component with complex dtype
        (``Complex[Array, "*shape"]``).
    """
    kz2 = jnp.asarray(eps) * omega**2 - kpar**2
    kz = jnp.sqrt(kz2.astype(complex))
    # If kz is complex we are not guaranteed to be in the right branch
    # but physically we need Im(kz) >= 0 to enforce decay.
    # So we flip the sign if necessary.
    kz = jnp.where(jnp.imag(kz) < 0, -kz, kz)
    return kz


@jaxtyped(typechecker=beartype)
def fresnel_interface(
    eps1: Complex[Array, "..."] | complex,
    eps2: Complex[Array, "..."] | complex,
    kz1: Complex[Array, "..."],
    kz2: Complex[Array, "..."],
    pol: Polarization,
) -> Tuple[Complex[Array, "..."], Complex[Array, "..."]]:
    """Compute Fresnel coefficients at a single interface (1 -> 2).

    Corresponds to Eq. 18 in the paper by Polder & Van Hove.

    Parameters
    ----------
    eps1, eps2 : complex or array
        Permittivities of media 1 and 2 (broadcastable).
    kz1, kz2 : complex array
        Normal wavevector components in media 1 and 2.
    pol : {"s", "p"}
        Polarization: ``"s"`` (TE) or ``"p"`` (TM).

    Returns
    -------
    r12 : complex array
        Reflection amplitude coefficient.
    t12 : complex array
        Transmission amplitude coefficient.
    """
    if pol == "s":
        r = (kz1 - kz2) / (kz1 + kz2)
        t = 2 * kz1 / (kz1 + kz2)
    elif pol == "p":
        r = (eps2 * kz1 - eps1 * kz2) / (eps2 * kz1 + eps1 * kz2)
        t = 2 * jnp.sqrt(eps1 * eps2) * kz1 / (eps2 * kz1 + eps1 * kz2)
    else:
        raise ValueError(f"Invalid polarization: {pol}")
    return r, t


@jaxtyped(typechecker=beartype)
def halfspace_RT(
    eps_halfspace: Complex[Array, "..."] | complex,
    omega: Float[Array, "..."] | float,
    kpar: Float[Array, "..."] | float,
    pol: Polarization,
) -> Tuple[Complex[Array, "..."], Complex[Array, "..."]]:
    r"""Reflection and transmission amplitudes of a semi-infinite half-space.

    Geometry: vacuum | half-space.

    Parameters
    ----------
    eps_halfspace : complex or array
        Half-space permittivity.
    omega : float or array
        Angular frequency.
    kpar : float or array
        In-plane wavevector magnitude $k_{\parallel}$.
    pol : {"s", "p"}
        Polarization: ``"s"`` (TE) or ``"p"`` (TM).

    Returns
    -------
    R : complex array
        Complex reflection amplitude coefficient as seen from vacuum.
    T : complex array
        Transmission amplitude coefficient through the full body. This is
        identically zero for a semi-infinite body because there is no second
        vacuum interface.
    """
    eps_vac = 1.0 + 0j
    kz0 = _kz(eps_vac, omega, kpar)
    kz_halfspace = _kz(eps_halfspace, omega, kpar)
    reflection, _ = fresnel_interface(eps_vac, eps_halfspace, kz0, kz_halfspace, pol)
    transmission = jnp.zeros_like(reflection)
    return reflection, transmission


@jaxtyped(typechecker=beartype)
def slab_RT(
    eps_slab: Complex[Array, "..."] | complex,
    omega: Float[Array, "..."] | float,
    kpar: Float[Array, "..."] | float,
    thickness: Float[Array, "..."] | float | None,
    pol: Polarization,
) -> Tuple[Complex[Array, "..."], Complex[Array, "..."]]:
    r"""Reflection and transmission amplitudes of a planar body.

    Corresponds to Eq. 23 in the paper by Polder & Van Hove for finite slabs,
    and reduces to the single-interface Fresnel reflection for a semi-infinite
    half-space.

    Geometry:

    - finite slab: vacuum | slab | vacuum
    - semi-infinite body: vacuum | half-space

    Parameters
    ----------
    eps_slab : complex or array
        Body permittivity.
    omega : float or array
        Angular frequency.
    kpar : float or array
        In-plane wavevector magnitude $k_{\parallel}$.
    thickness : float, array, or None
        Body thickness. Pass ``None`` for a semi-infinite half-space.
    pol : {"s", "p"}
        Polarization: ``"s"`` (TE) or ``"p"`` (TM).

    Returns
    -------
    R : complex array
        Complex reflection amplitude coefficient.
    T : complex array
        Complex transmission amplitude coefficient through the full body.

    Notes
    -----
    For ``thickness is None``, the body is treated as a semi-infinite passive
    medium and the result is ``(r_01, 0)``.

    For finite ``thickness``, the result corresponds to a Fabry--Perot slab
    embedded in vacuum ($\varepsilon = 1$).
    """
    if thickness is None:
        return halfspace_RT(eps_slab, omega, kpar, pol)

    eps_vac = 1.0 + 0j
    kz0 = _kz(eps_vac, omega, kpar)
    kzs = _kz(eps_slab, omega, kpar)

    r01, t01 = fresnel_interface(eps_vac, eps_slab, kz0, kzs, pol)
    r10, t10 = fresnel_interface(eps_slab, eps_vac, kzs, kz0, pol)

    phase = jnp.exp(1j * kzs * thickness)
    denom = 1 - r10 * r10 * phase**2

    R = r01 + t01 * r10 * phase**2 * t10 / denom
    T = t01 * phase * t10 / denom
    return R, T


@jaxtyped(typechecker=beartype)
def polder_van_hove_per_mode(
    R_A: Complex[Array, "..."],
    T_A: Complex[Array, "..."],
    R_B: Complex[Array, "..."],
    T_B: Complex[Array, "..."],
    kz0: Complex[Array, "..."],
    gap: Float[Array, "..."] | float,
) -> Float[Array, "..."]:
    r"""Compute the Polder-Van Hove transmission for one scalar mode.

    Corresponds to Eq. 23 and 25 in the paper by Polder & Van Hove.

    Parameters
    ----------
    R_A, T_A : complex array
        Reflection and transmission amplitudes of body A (as seen from the gap).
    R_B, T_B : complex array
        Reflection and transmission amplitudes of body B (as seen from the gap).
    kz0 : complex array
        Vacuum normal wavevector component.
    gap : float or array
        Vacuum gap thickness.

    Returns
    -------
    T_mode : float array
        Dimensionless transmission (real-valued).

    Notes
    -----
    For propagating modes ($\Re(k_z) > 0$, $\Im(k_z) \approx 0$):

    $$T = \frac{(1-|R_A|^2-|T_A|^2)(1-|R_B|^2-|T_B|^2)}
    {|1 - R_A R_B e^{2 i k_z d}|^2}.$$

    For evanescent modes ($\Im(k_z) > 0$, $\Re(k_z) \approx 0$):

    $$T = \frac{4\,\Im(R_A)\,\Im(R_B)\,e^{-2\kappa d}}
    {|1 - R_A R_B e^{-2\kappa d}|^2}.$$
    """
    phase2 = jnp.exp(2j * kz0 * gap)
    denom = jnp.abs(1.0 - R_A * R_B * phase2) ** 2

    is_prop = jnp.imag(kz0) < 1e-10 * jnp.abs(kz0)

    # Propagating
    abs_A = 1 - jnp.abs(R_A) ** 2 - jnp.abs(T_A) ** 2
    abs_B = 1 - jnp.abs(R_B) ** 2 - jnp.abs(T_B) ** 2
    T_prop = abs_A * abs_B / denom

    # Evanescent
    T_evan = 4 * jnp.imag(R_A) * jnp.imag(R_B) * jnp.abs(phase2) / denom

    return jnp.where(is_prop, T_prop, T_evan)


@jaxtyped(typechecker=beartype)
def polder_van_hove_integrand(
    kpar: Float[Array, "..."] | float,
    omega: Float[Array, "..."] | float,
    eps_A: Complex[Array, "..."] | complex,
    thickness_A: Float[Array, "..."] | float | None,
    eps_B: Complex[Array, "..."] | complex,
    thickness_B: Float[Array, "..."] | float | None,
    gap: Float[Array, "..."] | float,
) -> Float[Array, "..."]:
    r"""Compute the 1D radial integrand for the PVH planar formula.

    Corresponds to Eq. 22 and 24 in the paper by Polder & Van Hove.

    Returns
    -------
    $$\frac{k_{\parallel}}{2\pi}\sum_{p \in \{s,p\}} T(k_{\parallel}, p).$$

    Parameters
    ----------
    kpar : float or array
        In-plane wavevector magnitude $k_{\parallel}$.
    omega : float or array
        Angular frequency.
    eps_A, eps_B : complex or array
        Permittivities of slabs A and B.
    thickness_A, thickness_B : float, array, or None
        Thicknesses of bodies A and B. Pass ``None`` for semi-infinite
        half-spaces.
    gap : float or array
        Vacuum gap thickness.

    Returns
    -------
    integrand : float array
        Real-valued integrand sampled at ``kpar``.
    """
    kz0 = _kz(1.0 + 0j, omega, kpar)
    total = jnp.zeros_like(kpar, dtype=float)
    for pol in ("s", "p"):
        R_A, T_A = slab_RT(eps_A, omega, kpar, thickness_A, pol)
        R_B, T_B = slab_RT(eps_B, omega, kpar, thickness_B, pol)
        total += polder_van_hove_per_mode(R_A, T_A, R_B, T_B, kz0, gap)
    return kpar / (2 * jnp.pi) * total


@jaxtyped(typechecker=beartype)
def polder_van_hove_integrated(
    omega: Float[Array, "..."] | float,
    eps_A: Complex[Array, "..."] | complex,
    thickness_A: Float[Array, "..."] | float | None,
    eps_B: Complex[Array, "..."] | complex,
    thickness_B: Float[Array, "..."] | float | None,
    gap: Float[Array, "..."] | float,
    kpar_max_factor: float = 30.0,
    n_kpar: int = 8000,
) -> Float[Array, "..."]:
    r"""Integrate the PVH planar transfer over in-plane wavevectors.

    Computes the azimuthally-symmetric 2D integral

    $$\int \frac{d^2k_{\parallel}}{(2\pi)^2}\sum_p T(k_{\parallel}, p)$$

    via a 1D radial integral using a uniform grid in $k_{\parallel}$.

    Parameters
    ----------
    omega : float or array
        Angular frequency.
    eps_A, eps_B : complex or array
        Permittivities of slabs A and B.
    thickness_A, thickness_B : float, array, or None
        Thicknesses of bodies A and B. Pass ``None`` for semi-infinite
        half-spaces.
    gap : float or array
        Vacuum gap thickness.
    kpar_max_factor : float, optional
        Sets $k_{\parallel,\max} \approx \mathrm{kpar\_max\_factor}/\mathrm{gap}$.
    n_kpar : int, optional
        Number of $k_{\parallel}$ samples.

    Returns
    -------
    transfer_int : float array
        Integrated dimensionless transfer.

    Notes
    -----
    The spectral heat flux per unit area is typically assembled as

    $$\Phi(\omega)/A = \Theta(\omega, T) \times \mathrm{transfer\_int}.$$
    """
    kpar_max = kpar_max_factor / gap + 2 * omega
    kpar = jnp.linspace(1e-12, kpar_max, n_kpar)
    dk = kpar[1] - kpar[0]
    integrand = polder_van_hove_integrand(
        kpar, omega, eps_A, thickness_A, eps_B, thickness_B, gap
    )
    return jnp.sum(integrand) * dk


@jaxtyped(typechecker=beartype)
def planar_spectral_flux(
    omega: Float[Array, "*shape"],
    eps_A: Complex[Array, "*shape"],
    eps_B: Complex[Array, "*shape"],
    gap: float,
    theta_hot: Float[Array, "*shape"],
    theta_cold: Float[Array, "*shape"],
    thickness_A: float | None = None,
    thickness_B: float | None = None,
    kpar_max_factor: float = 30.0,
    n_kpar: int = 8000,
) -> Float[Array, "*shape"]:
    r"""Compute the planar spectral heat flux between two bodies.

    Batches the Polder-Van Hove integration over an array of frequencies
    and corresponding permittivities.

    Parameters
    ----------
    omega : array
        Angular frequencies.
    eps_A, eps_B : complex arrays
        Frequency-dependent permittivities for bodies A and B.
    gap : float
        Vacuum gap thickness.
    theta_hot, theta_cold : arrays
        Thermal energy terms (e.g., mean Bose-Einstein distributions) for
        the hot and cold bodies.
    thickness_A, thickness_B : float or None, optional
        Thicknesses of the bodies. Pass None for semi-infinite.
    kpar_max_factor : float, optional
        Sets the maximum in-plane wavevector integration limit.
    n_kpar : int, optional
        Number of k_parallel samples for the integration.

    Returns
    -------
    phi_omega : float array
        The spectral heat flux array matching the shape of omega.
    """
    transfer_1d = jax.vmap(
        polder_van_hove_integrated, in_axes=(0, 0, None, 0, None, None, None, None)
    )(omega, eps_A, thickness_A, eps_B, thickness_B, gap, kpar_max_factor, n_kpar)

    return omega * (theta_hot - theta_cold) * transfer_1d


@jaxtyped(typechecker=beartype)
def frequency_integrated_planar_spectral_flux(
    omega: Float[Array, "*shape"],
    eps_A: Complex[Array, "*shape"],
    eps_B: Complex[Array, "*shape"],
    gap: float,
    theta_hot: Float[Array, "*shape"],
    theta_cold: Float[Array, "*shape"],
    thickness_A: float | None = None,
    thickness_B: float | None = None,
    kpar_max_factor: float = 30.0,
    n_kpar: int = 8000,
) -> Float[Array, ""]:
    r"""Compute the total frequency-integrated planar heat flux.

    Calculates the spectral heat flux and integrates it over the frequency
    array `omega` using the trapezoidal rule.

    Parameters
    ----------
    omega : array
        Angular frequencies.
    eps_A, eps_B : complex arrays
        Frequency-dependent permittivities for bodies A and B.
    gap : float
        Vacuum gap thickness.
    theta_hot, theta_cold : arrays
        Thermal energy terms (e.g., mean Bose-Einstein distributions).
    thickness_A, thickness_B : float or None, optional
        Thicknesses of the bodies. Pass None for semi-infinite.
    kpar_max_factor : float, optional
        Sets the maximum in-plane wavevector integration limit.
    n_kpar : int, optional
        Number of k_parallel samples for the integration.

    Returns
    -------
    total_flux : scalar float array
        The total integrated heat flux (e.g., in W/m^2).
    """
    phi_omega = planar_spectral_flux(
        omega=omega,
        eps_A=eps_A,
        eps_B=eps_B,
        gap=gap,
        theta_hot=theta_hot,
        theta_cold=theta_cold,
        thickness_A=thickness_A,
        thickness_B=thickness_B,
        kpar_max_factor=kpar_max_factor,
        n_kpar=n_kpar,
    )

    return jnp.trapezoid(phi_omega, x=omega)


### Trace formulas ###


@jaxtyped(typechecker=beartype)
def transfer_per_mode(
    R_A: Complex[Array, "..."],
    T_A: Complex[Array, "..."],
    R_B: Complex[Array, "..."],
    T_B: Complex[Array, "..."],
    kz0: Complex[Array, "..."],
    gap: Float[Array, "..."] | float,
    omega: Float[Array, "..."] | float,
) -> Float[Array, "..."]:
    r"""Compute the scalar trace-formula transfer for one channel.

    This evaluates the scalar analogue of the matrix trace formula,

    $$\mathrm{Tr}_{\mathrm{single}} = P^* D^*\,\Sigma_A\,D\,P\,
    \frac{\Sigma_B}{|k_{z0}|^2},$$

    which should equal the standard PVH transmission for a single scalar mode.

    Parameters
    ----------
    R_A, T_A : complex array
        Reflection and transmission amplitudes of body A.
    R_B, T_B : complex array
        Reflection and transmission amplitudes of body B.
    kz0 : complex array
        Vacuum normal wavevector component.
    gap : float or array
        Vacuum gap thickness.
    omega : float or array
        Angular frequency.

    Returns
    -------
    transfer : float array
        Real-valued dimensionless transfer for this channel.

    Notes
    -----
    For a single mode, the absorption operator used here is

    $$\Sigma = \Re(k_{z0})(1 - |R|^2 - |T|^2) - 2\,\Im(k_{z0})\,\Im(R).$$
    """
    P = jnp.exp(1j * kz0 * gap)
    denom = 1.0 - P * R_B * P * R_A
    D = 1.0 / denom

    kz_re: Float[Array, "*shape"] = jnp.real(kz0)
    kz_im: Float[Array, "*shape"] = jnp.imag(kz0)

    def sigma(
        R: Complex[Array, "*shape"],
        T: Complex[Array, "*shape"],
    ) -> Float[Array, "*shape"]:
        return kz_re * (1 - jnp.abs(R) ** 2 - jnp.abs(T) ** 2) - 2 * kz_im * jnp.imag(R)

    sig_A = sigma(R_A, T_A)
    sig_B = sigma(R_B, T_B)

    P_dag = jnp.conj(P)
    D_dag = jnp.conj(D)

    # scalar trace: P* D* Σ_A D P Σ_B / |kz0|²
    kz_abs_sq = kz_re**2 + kz_im**2
    return jnp.real(P_dag * D_dag * sig_A * D * P * sig_B) / kz_abs_sq


@jaxtyped(typechecker=beartype)
def transfer_kpar_integrand(
    kpar: Float[Array, "..."] | float,
    omega: Float[Array, "..."] | float,
    eps_A: Complex[Array, "..."] | complex,
    thickness_A: Float[Array, "..."] | float | None,
    eps_B: Complex[Array, "..."] | complex,
    thickness_B: Float[Array, "..."] | float | None,
    gap: Float[Array, "..."] | float,
) -> Float[Array, "..."]:
    r"""Compute the radial integrand for the scalar trace transfer.

    For the planar azimuthally-symmetric case,

    $$\int \frac{d^2k_{\parallel}}{(2\pi)^2} T(k_{\parallel})
    = \int_0^\infty dk_{\parallel}\,\frac{k_{\parallel}}{2\pi}
    \sum_p T(k_{\parallel}, p).$$

    Parameters
    ----------
    kpar : float or array
        In-plane wavevector magnitude $k_{\parallel}$.
    omega : float or array
        Angular frequency.
    eps_A, eps_B : complex or array
        Permittivities of slabs A and B.
    thickness_A, thickness_B : float, array, or None
        Thicknesses of bodies A and B. Pass ``None`` for semi-infinite
        half-spaces.
    gap : float or array
        Vacuum gap thickness.

    Returns
    -------
    integrand : float array
        Real-valued integrand sampled at ``kpar``.
    """
    kz0 = _kz(1.0 + 0j, omega, kpar)
    total = jnp.zeros_like(kpar, dtype=float)
    for pol in ("s", "p"):
        R_A, T_A = slab_RT(eps_A, omega, kpar, thickness_A, pol)
        R_B, T_B = slab_RT(eps_B, omega, kpar, thickness_B, pol)
        total += transfer_per_mode(R_A, T_A, R_B, T_B, kz0, gap, omega)
    return kpar / (2 * jnp.pi) * total


@jaxtyped(typechecker=beartype)
def integrated_transfer(
    omega: Float[Array, "..."] | float,
    eps_A: Complex[Array, "..."] | complex,
    thickness_A: Float[Array, "..."] | float | None,
    eps_B: Complex[Array, "..."] | complex,
    thickness_B: Float[Array, "..."] | float | None,
    gap: Float[Array, "..."] | float,
    kpar_max_factor: float = 30.0,
    n_kpar: int = 4000,
) -> Float[Array, "..."]:
    r"""Integrate the scalar trace transfer over in-plane wavevectors.

    Performs a uniform-grid approximation of

    $$\int \frac{d^2k_{\parallel}}{(2\pi)^2}\sum_p T(k_{\parallel}, p),$$

    integrating from $k_{\parallel}=0$ up to a cutoff chosen to capture
    evanescent contributions.

    Parameters
    ----------
    omega : float or array
        Angular frequency.
    eps_A, eps_B : complex or array
        Permittivities of slabs A and B.
    thickness_A, thickness_B : float, array, or None
        Thicknesses of bodies A and B. Pass ``None`` for semi-infinite
        half-spaces.
    gap : float or array
        Vacuum gap thickness.
    kpar_max_factor : float, optional
        Sets $k_{\parallel,\max} \approx \mathrm{kpar\_max\_factor}/\mathrm{gap}$.
    n_kpar : int, optional
        Number of $k_{\parallel}$ samples.

    Returns
    -------
    transfer_int : float array
        Integrated dimensionless transfer.
    """
    kpar_max = kpar_max_factor / gap + 2 * omega
    kpar = jnp.linspace(1e-12, kpar_max, n_kpar)
    dk = kpar[1] - kpar[0]
    integrand = transfer_kpar_integrand(
        kpar, omega, eps_A, thickness_A, eps_B, thickness_B, gap
    )
    return jnp.sum(integrand) * dk
