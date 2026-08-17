#!/usr/bin/env python3
"""Maximize single-frequency heat transfer between two slabs by tuning a hole diameter.

    [vac]  |  slab (holes)  |  gap d  |  slab (holes)  |  [vac]

Both bodies share the same patterned slab. The Brillouin-zone average is evaluated
through ``compute_bz_average``, which chunks the k-points; substitute your own
``kernel(omega_i, eps_i, k_points_chunk) -> (n_k_chunk,)`` for a different geometry,
using ``kernels.two_body_tau_kernel`` as the reference implementation.
"""

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

from fluxmax.optimization.design_tools import circular_exclusion_permittivity
from fluxmax.parallelism import compute_bz_average, flatten_k_points
from fluxmax.physics import heat_transfer as ht
from fluxmax.physics.kernels import make_two_body_bz_kernel
from fluxmax.setup import two_body as ss

WAVELENGTH = 1.0
PITCH = 0.93  # not 1.0: a commensurate pitch puts grazing orders on the light line
SLAB_THICKNESS = 0.5
GAP = 0.2
EPS_SLAB = 2.25 + 0.5j
EPS_HOLE = 1.0 + 0.0j
SOFTNESS = 0.01  # nonzero so the boundary has a gradient

APPROXIMATE_NUM_TERMS = 25
BZ_GRID = (5, 5)
K_CHUNK_SIZE = 5
RESOLUTION = PITCH / 512

OUTPUT_DIR = "single_frequency_output"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--diameter-init", type=float, default=0.3)
    parser.add_argument("--grad-check", action="store_true",
                        help="Compare the AD gradient against central differences.")
    parser.add_argument("--out", default=OUTPUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    plv, expansion, in_plane_wavevector = ss.make_rcwa_setup(
        pitch=PITCH,
        approximate_num_terms=APPROXIMATE_NUM_TERMS,
        brillouin_grid_shape=BZ_GRID,
    )
    k_points = flatten_k_points(in_plane_wavevector)
    area = ss.cell_area(plv)
    omega = float(ht.wavelength_to_omega(jnp.asarray(WAVELENGTH)))

    print("JAX devices:", jax.devices())
    print(f"Fourier terms: {expansion.num_terms}, BZ points: {k_points.shape[0]}")
    print(f"omega: {omega:.4f}, cell area: {float(area):.4f}")

    kernel = make_two_body_bz_kernel(
        primitive_lattice_vectors=plv,
        expansion=expansion,
        slab_thickness=jnp.asarray(SLAB_THICKNESS),
        gap=jnp.asarray(GAP),
    )

    def permittivity(diameter: jnp.ndarray) -> jnp.ndarray:
        return circular_exclusion_permittivity(
            pitch=PITCH,
            diameter=diameter,
            eps_slab=EPS_SLAB,
            eps_exclusion=EPS_HOLE,
            resolution=RESOLUTION,
            softness=SOFTNESS,
        )

    def transfer_from_diameter(diameter: jnp.ndarray) -> jnp.ndarray:
        """BZ-averaged tau per unit area."""
        tau_avg = compute_bz_average(
            kernel,
            omega_1d=jnp.asarray([omega]),
            eps_omega=permittivity(diameter)[jnp.newaxis, ...],
            k_points=k_points,
            execution_mode="single_device_chunked",
            k_chunk_size=K_CHUNK_SIZE,
            omega_chunk_size=1,
        )
        return tau_avg[0] / area

    if args.grad_check:
        # eigensolve_patterned defaults to Formulation.FFT, whose gradient is exact;
        # the vector formulations stop_gradient the tangent field and would not match.
        d_test = jnp.asarray(0.4)
        ad = float(jax.grad(transfer_from_diameter)(d_test))
        h = 1e-5
        fd = float(
            (transfer_from_diameter(d_test + h) - transfer_from_diameter(d_test - h))
            / (2 * h)
        )
        print(f"d(tau/area)/d(diameter): AD {ad:.6e}  FD {fd:.6e}  "
              f"rel {abs(ad - fd) / abs(fd):.2e}")

    def diameter_from_param(t: jnp.ndarray) -> jnp.ndarray:
        """Keep d in (0, pitch) for unconstrained t."""
        return PITCH * jax.nn.sigmoid(t)

    def loss(t: jnp.ndarray) -> jnp.ndarray:
        return -transfer_from_diameter(diameter_from_param(t))

    loss_and_grad = jax.jit(jax.value_and_grad(loss))

    t = jnp.asarray(float(np.log(args.diameter_init / (PITCH - args.diameter_init))))
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(t)
    history: dict[str, list[float]] = {"step": [], "diameter": [], "transfer": []}

    print(f"\n{'Step':>5}  {'diameter':>10}  {'tau / area':>14}")
    print("-" * 33)
    for step in range(args.steps):
        loss_value, grads = loss_and_grad(t)
        updates, opt_state = optimizer.update(grads, opt_state)
        t = optax.apply_updates(t, updates)

        history["step"].append(step)
        history["diameter"].append(float(diameter_from_param(t)))
        history["transfer"].append(float(-loss_value))
        if step % 5 == 0 or step == args.steps - 1:
            print(f"{step:>5}  {history['diameter'][-1]:>10.4f}  "
                  f"{history['transfer'][-1]:>14.6e}")

    d_opt = history["diameter"][-1]
    print(f"\nOptimal diameter: {d_opt:.4f} (units of pitch)")
    print(f"Max transfer / area: {history['transfer'][-1]:.6e}")

    np.savez(
        os.path.join(args.out, "circular_exclusion_history.npz"),
        **{k: np.asarray(v) for k, v in history.items()},
        pitch=PITCH, gap=GAP, slab_thickness=SLAB_THICKNESS, omega=omega,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["step"], history["transfer"], linewidth=2)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Transfer / area")
    ax.grid(True)
    fig.tight_layout()
    history_path = os.path.join(args.out, "circular_exclusion_history.png")
    fig.savefig(history_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(
        np.real(np.asarray(permittivity(jnp.asarray(d_opt)))),
        origin="lower",
        cmap="viridis",
    )
    ax.set_title(f"Re(eps) at d* = {d_opt:.3f}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    pattern_path = os.path.join(args.out, "circular_exclusion_pattern.png")
    fig.savefig(pattern_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {history_path} and {pattern_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
