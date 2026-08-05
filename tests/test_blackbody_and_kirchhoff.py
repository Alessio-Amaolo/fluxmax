"""Absolute scale: blackbody prefactors and Kirchhoff, against outside references.

Most of the suite is relative. PVH agreement fixes the planar transfer but says
nothing about mode-mixing bodies; reciprocity and plane-shift invariance are
self-consistency statements about the same matrices, so a normalization error common
to both sides survives them.

1. Kirchhoff, per Bloch sector.

Put a black absorber (R = T = 0) in front of a patterned emitter: D = I, the absorber's
Sigma is the flux form F_re which is supported on propagating channels:

    tau = Tr[P+ F_re P (F+)^-1 Sigma_emitter^T F^-1]

which is gap-independent, because F is block diagonal in G and the two
polarizations of a block share k_z, so P commutes with it. It must equal the
emitter's absorptivity summed over propagating channels.

The reference is measured with fmmax's Poynting routine on an explicitly built
flux-orthonormal basis of the propagating subspace. This is the only
independent handle on the emitter normalization |K|^-1 Sigma^T |K|^-1 for a
mode-mixing body. Caveat: a black absorber probes only the *propagating* block; the
evanescent normalization is pinned by the split-PVH tests in test_planar_pvh and by
test_reference_plane_shift.

2. The prefactor chain, trace -> W/m^2 -> Stefan-Boltzmann.

With both bodies black the trace is exactly the propagating channel count, so the
notes' blackbody check applies: the BZ integral is omega^2 / (2 pi), and hbar omega Theta / (2 pi) times
that integrates to sigma T^4.
"""

import jax.numpy as jnp
import numpy as np
from _helpers import K_GENERIC, centered_grating, net_flux, squeeze_batch

from fluxmax.physics import heat_transfer as ht
from fluxmax.setup import two_body as ss
from fluxmax.units import si_units

WAVELENGTH = 1.0
PITCH = 1.0
OMEGA = 2.0 * np.pi / WAVELENGTH
K_POINT = K_GENERIC[np.newaxis, :]

EMITTER_THICKNESS = 0.4
EPS_STRIPE = 12.0 + 0.5j
FILL = 0.4
TERMS = [1, 9, 21]
GAPS = [0.2, 0.7, 3.0]
KIRCHHOFF_TOL = 1e-9
GAP_INDEPENDENCE_TOL = 1e-12
MIN_WRONG_GAUGE_ERROR = 1e-2

T_NAT = 0.5
OMEGA_MIN, OMEGA_MAX = 0.02 * T_NAT, 25.0 * T_NAT
N_OMEGA = 120
BZ_N = 181
BZ_SPAN = 1.5  # BZ half-width in units of the light-cone radius omega
SIGMA_SB_NAT = np.pi**2 / 60.0  # Stefan-Boltzmann constant with hbar=c=k_B=1
CHANNEL_COUNT_TOL = 1e-10
BZ_QUADRATURE_TOL = 2e-3
STEFAN_BOLTZMANN_TOL = 5e-3
EXACT_TOL = 1e-12


# ==========================================================================
# Kirchhoff's law per Bloch sector
# ==========================================================================


def _build_emitter(num_terms: int):
    """Emitter = a lossy grating in the ``sigma_B`` (emitter) slot; absorber =
    black body."""
    plv, expansion, _ = ss.make_rcwa_setup(pitch=PITCH, approximate_num_terms=num_terms)
    kw = dict(
        wavelength=jnp.asarray(WAVELENGTH),
        in_plane_wavevector=jnp.asarray(K_POINT),
        primitive_lattice_vectors=plv,
        expansion=expansion,
    )
    vac = ss.eigensolve_uniform(**kw, permittivity=1.0 + 0.0j)
    slab = ss.eigensolve_patterned(
        **kw, permittivity_array=centered_grating(EPS_STRIPE, FILL)
    )
    R_e, T_e, _ = ss.body_s_matrices(
        vac, slab, jnp.asarray(EMITTER_THICKNESS), is_body_A=False
    )
    F_re, F_ah, F = ht.poynting_flux_matrices(vac)
    sigma_e = ht.compute_sigma(R_e, T_e, F_re, F_ah)
    zero = jnp.zeros_like(jnp.asarray(R_e))
    sigma_black = ht.compute_sigma(zero, zero, F_re, F_ah)
    return dict(
        vac=vac,
        R_e=R_e,
        T_e=T_e,
        F_re=F_re,
        F=F,
        sigma_e=sigma_e,
        sigma_black=sigma_black,
        zero=zero,
    )


def _propagating_indices(vac) -> np.ndarray:
    q = np.asarray(vac.eigenvalues)
    q = q[0] if q.ndim == 2 else q
    return np.where(np.abs(q.imag) < 1e-9 * np.abs(q))[0]


def _flux_orthonormal_incidences(vac, F_re) -> np.ndarray:
    """Columns spanning the propagating subspace, each carrying unit incident
    flux and mutually flux-orthogonal, so that summing their absorptivities is
    a basis-independent trace."""
    F_re = np.asarray(squeeze_batch(F_re))
    idx = _propagating_indices(vac)

    outside = F_re.copy()
    outside[np.ix_(idx, idx)] = 0.0
    assert np.linalg.norm(outside) < 1e-12 * np.linalg.norm(F_re), (
        "F_re has support outside the propagating block; the projection used "
        "for the Kirchhoff reference is not valid"
    )

    block = F_re[np.ix_(idx, idx)]
    assert np.max(np.abs(block.imag)) < 1e-12 * np.max(np.abs(block))
    chol = np.linalg.cholesky(block)  # block = L L^dag, positive definite
    weights = np.linalg.solve(chol.conj().T, np.eye(len(idx)))
    basis = np.zeros((F_re.shape[0], len(idx)), dtype=complex)
    basis[idx, np.arange(len(idx))] = 1.0
    return np.asarray(basis @ weights)


def _absorptivities(vac, R, T, incidences) -> np.ndarray:
    """Absorptivity of the emitter for each incident channel, from fmmax's own
    Poynting routine."""
    R, T = np.asarray(squeeze_batch(R)), np.asarray(squeeze_batch(T))
    out = []
    for i in range(incidences.shape[1]):
        b = incidences[:, i]
        incident = net_flux(vac, b, np.zeros_like(b))
        absorbed = net_flux(vac, b, R @ b) - net_flux(vac, T @ b, np.zeros_like(b))
        out.append(absorbed / incident)
    return np.array(out)


def _tau_black_absorber(m, gap: float) -> float:
    P = ht.propagation_matrix(m["vac"].eigenvalues, gap)
    tau = ht.spectral_transfer(
        m["sigma_black"], m["sigma_e"], P, m["zero"], m["R_e"], m["F"]
    )
    return float(np.real(np.asarray(tau)).ravel()[0])


def test_emissivity_equals_absorptivity_for_a_grating():
    for num_terms in TERMS:
        m = _build_emitter(num_terms)
        incidences = _flux_orthonormal_incidences(m["vac"], m["F_re"])
        absorptivity = _absorptivities(m["vac"], m["R_e"], m["T_e"], incidences)

        assert absorptivity.min() > 0.0, (
            f"num_terms={num_terms}: non-absorbing channel {absorptivity.min():.3e}"
        )
        assert absorptivity.max() < 1.0, (
            f"num_terms={num_terms}: absorptivity above unity {absorptivity.max():.3e}"
        )

        reference = float(absorptivity.sum())
        tau = _tau_black_absorber(m, gap=GAPS[0])
        rel = abs(tau - reference) / abs(reference)
        assert rel < KIRCHHOFF_TOL, (
            f"num_terms={num_terms} ({len(absorptivity)} propagating "
            f"channels): tau={tau:.10e} vs summed absorptivity "
            f"{reference:.10e}, rel={rel:.3e}"
        )


def test_black_absorber_transfer_is_gap_independent():
    """With R = T = 0 on the absorber only propagating channels contribute, and
    the gap propagator is unitary on them."""
    for num_terms in TERMS:
        m = _build_emitter(num_terms)
        values = [_tau_black_absorber(m, gap=gap) for gap in GAPS]
        spread = (max(values) - min(values)) / abs(values[0])
        assert spread < GAP_INDEPENDENCE_TOL, (
            f"num_terms={num_terms}: tau varies with gap by {spread:.3e} "
            f"({dict(zip(GAPS, values))})"
        )


def test_kirchhoff_is_sensitive_to_the_emitter_normalization():
    """Negative control: replacing the F-gauge correlator normalization with the
    naive diag(1/|k_z|) of the notes' amplitude basis breaks Kirchhoff, so the
    check above is actually testing the normalization."""
    m = _build_emitter(9)
    incidences = _flux_orthonormal_incidences(m["vac"], m["F_re"])
    reference = float(_absorptivities(m["vac"], m["R_e"], m["T_e"], incidences).sum())

    q = np.asarray(m["vac"].eigenvalues)
    q = q[0] if q.ndim == 2 else q
    inv_abs_k = np.diag(1.0 / np.abs(q))
    sigma_e = np.asarray(squeeze_batch(m["sigma_e"]))
    F_re = np.asarray(squeeze_batch(m["F_re"]))
    tau_wrong = float(np.real(np.trace(F_re @ inv_abs_k @ sigma_e.T @ inv_abs_k)))

    rel = abs(tau_wrong - reference) / abs(reference)
    assert rel > MIN_WRONG_GAUGE_ERROR, (
        "the |k_z|-gauge correlator reproduces Kirchhoff too, so this test "
        f"does not constrain the normalization (rel={rel:.3e})"
    )


# ==========================================================================
# The prefactor chain: trace -> W/m^2 -> Stefan-Boltzmann
# ==========================================================================


def _blackbody(omega: float, bz_n: int = BZ_N):
    """Per-k blackbody transfer at one frequency, with the BZ scaled to the
    light cone. Returns (tau, |k|, cell_area, n_bz).

    The pitch is scaled with omega so the first Brillouin zone is always
    ``BZ_SPAN`` light cones wide, which holds the quadrature error of the sharp
    light-cone boundary at a fixed ~5e-4 relative instead of blowing up at small
    omega.
    """
    pitch = np.pi / (BZ_SPAN * omega)
    plv, expansion, ipw = ss.make_rcwa_setup(
        pitch=pitch, approximate_num_terms=1, brillouin_grid_shape=(bz_n, bz_n)
    )
    k_points = jnp.reshape(ipw, (-1, 2))
    vac = ss.eigensolve_uniform(
        wavelength=jnp.asarray(2.0 * np.pi / omega),
        in_plane_wavevector=k_points,
        primitive_lattice_vectors=plv,
        expansion=expansion,
        permittivity=1.0 + 0.0j,
    )
    F_re, F_ah, F = ht.poynting_flux_matrices(vac)
    zero = jnp.zeros_like(F_re)
    sigma = ht.compute_sigma(zero, zero, F_re, F_ah)
    P = ht.propagation_matrix(vac.eigenvalues, 0.3)
    tau = np.real(np.asarray(ht.spectral_transfer(sigma, sigma, P, zero, zero, F)))
    return (
        tau,
        np.linalg.norm(np.asarray(k_points), axis=-1),
        float(np.asarray(ss.cell_area(plv))),
        bz_n * bz_n,
    )


def test_blackbody_trace_counts_propagating_channels():
    """R = T = 0 makes the trace the number of propagating channels, exactly."""
    for omega in (0.4, 2.5, 6.0):
        tau, kpar, _, _ = _blackbody(omega, bz_n=61)
        expected = 2.0 * (kpar < omega)  # two polarizations inside the cone
        worst = float(np.max(np.abs(tau - expected)))
        assert worst < CHANNEL_COUNT_TOL, (
            f"omega={omega}: blackbody tau is not the propagating channel "
            f"count, worst deviation {worst:.3e}"
        )
        assert expected.sum() > 0, "no propagating channels sampled"


def test_blackbody_bz_integral_is_omega_squared_over_two_pi():
    """The notes' blackbody consistency check, including the BZ measure."""
    for omega in (0.4, 2.5, 6.0):
        tau, _, area, n_bz = _blackbody(omega)
        bz_integral = float(np.sum(tau)) / (n_bz * area)
        expected = omega**2 / (2.0 * np.pi)
        rel = abs(bz_integral - expected) / expected
        assert rel < BZ_QUADRATURE_TOL, (
            f"omega={omega}: BZ-integrated blackbody trace {bz_integral:.6e} vs "
            f"omega^2/(2 pi) = {expected:.6e}, rel {rel:.3e}"
        )

    # The residual is light-cone quadrature error: refining the grid improves it.
    def error(bz_n: int) -> float:
        tau, _, area, n_bz = _blackbody(2.5, bz_n=bz_n)
        exact = 2.5**2 / (2.0 * np.pi)
        return float(abs(float(np.sum(tau)) / (n_bz * area) - exact) / exact)

    coarse, fine = error(61), error(361)
    assert fine < coarse, (
        f"BZ quadrature error did not improve with the grid: {fine:.3e} (N=361) "
        f"vs {coarse:.3e} (N=61)"
    )


def test_stefan_boltzmann_law_end_to_end():
    """hbar omega Theta / 2 pi, the BZ measure and the SI conversion together
    reproduce sigma T^4."""
    omega = np.linspace(OMEGA_MIN, OMEGA_MAX, N_OMEGA)
    phi_omega = np.zeros_like(omega)
    for index, omega_value in enumerate(omega):
        tau, _, area, n_bz = _blackbody(float(omega_value))
        phi_omega[index] = float(
            ht.spectral_heat_flux(
                jnp.asarray(float(np.sum(tau))),
                jnp.asarray(float(omega_value)),
                jnp.asarray(T_NAT),
                cell_area=area,
                n_bz=n_bz,
            )
        )

    # Elementwise against the analytic Planck spectrum omega^3 Theta / (4 pi^2).
    theta = np.asarray(ht.bose_einstein(jnp.asarray(omega), T_NAT))
    planck = omega**3 * theta / (4.0 * np.pi**2)
    rel_spectrum = np.abs(phi_omega - planck) / planck
    assert rel_spectrum.max() < BZ_QUADRATURE_TOL, (
        f"spectral flux vs Planck: worst rel {rel_spectrum.max():.3e} at "
        f"omega={omega[int(np.argmax(rel_spectrum))]:.3f}"
    )

    flux_nat = float(np.trapezoid(phi_omega, omega))
    expected_nat = SIGMA_SB_NAT * T_NAT**4
    rel = abs(flux_nat - expected_nat) / expected_nat
    assert rel < STEFAN_BOLTZMANN_TOL, (
        f"Stefan-Boltzmann in natural units: got {flux_nat:.6e}, expected "
        f"{expected_nat:.6e}, rel {rel:.3e}"
    )

    # The same statement in SI, which additionally exercises si_units.
    flux_si = float(si_units.flux_per_area_nat_to_SI(flux_nat))
    T_kelvin = float(si_units.temperature_nat_to_K(T_NAT))
    sigma_si = (
        np.pi**2 * si_units.KB_SI**4 / (60.0 * si_units.HBAR_SI**3 * si_units.C_SI**2)
    )
    expected_si = sigma_si * T_kelvin**4
    rel_si = abs(flux_si - expected_si) / expected_si
    assert rel_si < STEFAN_BOLTZMANN_TOL, (
        f"Stefan-Boltzmann in SI: got {flux_si:.6e} W/m^2 at T={T_kelvin:.1f} K, "
        f"expected {expected_si:.6e} W/m^2, rel {rel_si:.3e}"
    )


def test_spectral_heat_flux_is_elementwise_in_frequency():
    """A batch of frequencies must be treated independently: the BZ sum belongs
    to the caller, not to a hidden reduction over the whole array."""
    transfer = jnp.asarray([1.0, 2.0, 3.0])
    omega = jnp.asarray([0.7, 1.3, 2.9])
    temperature = jnp.full_like(omega, 1.1)
    n_bz = 4
    area = 2.0

    batched = np.asarray(
        ht.spectral_heat_flux(transfer, omega, temperature, cell_area=area, n_bz=n_bz)
    )
    for index in range(3):
        single = float(
            ht.spectral_heat_flux(
                transfer[index],
                omega[index],
                temperature[index],
                cell_area=area,
                n_bz=n_bz,
            )
        )
        assert abs(batched[index] - single) < EXACT_TOL * abs(single), (
            f"element {index}: batched {batched[index]:.6e} != single {single:.6e}"
        )

    theta = np.asarray(ht.bose_einstein(omega, temperature))
    expected = (
        np.asarray(omega) * theta / (2.0 * np.pi) * np.asarray(transfer) / n_bz / area
    )
    assert np.allclose(batched, expected, rtol=EXACT_TOL)


def test_si_unit_round_trips():
    """The natural<->SI conversions used above are mutually consistent."""
    for value in (0.05, 1.0, 37.0):
        assert np.isclose(
            float(si_units.omega_phys_to_nat(si_units.omega_nat_to_phys(value))),
            value,
            rtol=EXACT_TOL,
        )
        assert np.isclose(
            float(si_units.temperature_K_to_nat(si_units.temperature_nat_to_K(value))),
            value,
            rtol=EXACT_TOL,
        )
        assert np.isclose(
            float(si_units.length_m_to_nat(si_units.length_nat_to_m(value))),
            value,
            rtol=EXACT_TOL,
        )
        # A wavelength in natural units against the matching angular frequency.
        omega = 2.0 * np.pi / value
        assert np.isclose(
            float(si_units.omega_nat_to_wavelength_um(omega)),
            value * si_units.L0_M_DEFAULT * 1e6,
            rtol=EXACT_TOL,
        )

    # hbar c / L0^3 (spectral) vs hbar c^2 / L0^4 (frequency-integrated): the two
    # must differ by exactly c / L0, the natural unit of angular frequency.
    ratio = float(si_units.flux_per_area_nat_to_SI(1.0)) / float(
        si_units.spectral_flux_density_nat_to_SI(1.0)
    )
    assert np.isclose(ratio, si_units.C_SI / si_units.L0_M_DEFAULT, rtol=EXACT_TOL), (
        f"flux conversions are inconsistent: ratio {ratio:.6e}"
    )
