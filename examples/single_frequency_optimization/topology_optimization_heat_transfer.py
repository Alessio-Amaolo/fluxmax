#!/usr/bin/env python3
"""Single-frequency topology optimization of the net heat flux between two slabs.

Both bodies carry the same freely-designed density, projected with a tanh filter whose
sharpness beta is raised in stages so the pattern binarizes gradually.

Converted from topology_optimization_heat_transfer.ipynb.
"""

# ruff: noqa: E402  jax x64 and the Agg backend must be set before the imports below

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

from fluxmax.optimization.design_tools import dielectric_eps_from_density, project_tanh
from fluxmax.parallelism import compute_bz_average, flatten_k_points
from fluxmax.physics import heat_transfer as ht
from fluxmax.physics.kernels import broadcast_slab_permittivity, make_two_body_bz_kernel
from fluxmax.setup import two_body as tb
from fluxmax.utils.plot_utils import plot_square_bz_points

WAVELENGTH = 1.0
PITCH = 0.93  # not 1.0: a commensurate pitch puts grazing orders on the light line
SLAB_THICKNESS = 0.5
GAP = 0.5
TEMP_A = 1.0
TEMP_B = 2.0  # body B is the emitter, so the net flux is B -> A
EPS_VOID = 1.0 + 0.0j
EPS_SOLID = 10.0 + 0.01j

APPROXIMATE_NUM_TERMS = 100
BZ_GRID = (10, 10)
DESIGN_N = 64
EXECUTION_MODE = "single_device_chunked"
K_CHUNK_SIZE = 1
OMEGA_CHUNK_SIZE = 1

BETA_STAGES = (0.5, 5.0, 10.0, 20.0)
ITER_STAGES = (30, 20, 20, 20)

OUTPUT_DIR = "single_frequency_output"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--iter-stages", type=int, nargs="+", default=list(ITER_STAGES),
        help=f"Steps per beta stage; beta stages are {list(BETA_STAGES)}",
    )
    parser.add_argument("--out", default=OUTPUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if len(args.iter_stages) != len(BETA_STAGES):
        parser.error(f"--iter-stages needs {len(BETA_STAGES)} values")

    plv, expansion, in_plane_wavevector = tb.make_rcwa_setup(
        pitch=PITCH,
        approximate_num_terms=APPROXIMATE_NUM_TERMS,
        brillouin_grid_shape=BZ_GRID,
    )
    k_points = flatten_k_points(in_plane_wavevector)
    area = tb.cell_area(plv)
    omega = jnp.asarray(ht.wavelength_to_omega(jnp.asarray(WAVELENGTH)))

    print("JAX devices:", jax.devices())
    print(f"Fourier terms: {expansion.num_terms}, BZ points: {k_points.shape[0]}")
    print(f"omega: {float(omega):.6f}, cell area: {float(area):.4f}")

    kernel = make_two_body_bz_kernel(
        primitive_lattice_vectors=plv,
        expansion=expansion,
        slab_thickness=jnp.asarray(SLAB_THICKNESS),
        gap=jnp.asarray(GAP),
        eps_gap=EPS_VOID,
    )

    # hbar omega Theta / 2pi, with the 1/2pi of the trace formula's prefactor
    theta_a = ht.bose_einstein(omega, jnp.asarray(TEMP_A))
    theta_b = ht.bose_einstein(omega, jnp.asarray(TEMP_B))
    net_prefactor = omega * (theta_b - theta_a) / (2.0 * jnp.pi)

    def projected(
        rho: jnp.ndarray, beta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        rho_hat = project_tanh(rho, beta)
        eps_grid = dielectric_eps_from_density(
            rho=rho_hat, eps_solid=EPS_SOLID, eps_void=EPS_VOID
        )
        return rho_hat, eps_grid

    def net_spectral_flux_per_area(rho: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
        _, eps_grid = projected(rho, beta)
        tau_bz_avg = compute_bz_average(
            kernel_fn=kernel,
            omega_1d=jnp.asarray([omega]),
            eps_omega=broadcast_slab_permittivity(eps_grid, 1),
            k_points=k_points,
            execution_mode=EXECUTION_MODE,
            k_chunk_size=K_CHUNK_SIZE,
            omega_chunk_size=OMEGA_CHUNK_SIZE,
        )
        return net_prefactor * tau_bz_avg[0] / area

    def loss(rho: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
        return -net_spectral_flux_per_area(rho, beta)

    loss_and_grad = jax.jit(jax.value_and_grad(loss))

    rho = jax.random.uniform(jax.random.key(args.seed), (DESIGN_N, DESIGN_N))
    print(f"Initial net flux / area: "
          f"{float(net_spectral_flux_per_area(rho, jnp.asarray(BETA_STAGES[0]))):.6e}")

    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(rho)
    history: dict[str, list[float]] = {"step": [], "beta": [], "net_flux": []}
    step = 0
    print(f"\n{'Step':>5}  {'beta':>6}  {'net flux / area':>16}")
    print("-" * 31)
    for stage_beta, stage_steps in zip(BETA_STAGES, args.iter_stages):
        beta = jnp.asarray(stage_beta)
        for _ in range(stage_steps):
            loss_value, grads = loss_and_grad(rho, beta)
            updates, opt_state = optimizer.update(grads, opt_state, rho)
            rho = optax.apply_updates(rho, updates)
            history["step"].append(step)
            history["beta"].append(float(beta))
            history["net_flux"].append(float(-loss_value))
            if step % 5 == 0:
                flux = history["net_flux"][-1]
                print(f"{step:>5}  {float(beta):>6.2f}  {flux:>16.6e}")
            step += 1

    beta_final = jnp.asarray(BETA_STAGES[-1])
    rho_hat, eps_final = projected(rho, beta_final)
    final_flux = float(net_spectral_flux_per_area(rho, beta_final))
    print(f"\nFinal net flux / area: {final_flux:.6e}")

    np.savez(
        os.path.join(args.out, "topology_history.npz"),
        **{k: np.asarray(v) for k, v in history.items()},
        rho_hat=np.asarray(rho_hat), eps_final=np.asarray(eps_final),
        pitch=PITCH, gap=GAP, slab_thickness=SLAB_THICKNESS,
        omega=float(omega), final_flux=final_flux,
    )

    bz_path = os.path.join(args.out, "topology_bz_points.png")
    fig = plot_square_bz_points(in_plane_wavevector, PITCH)
    fig.savefig(bz_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["step"], history["net_flux"], linewidth=2)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Net flux / area")
    ax.grid(True)
    fig.tight_layout()
    history_path = os.path.join(args.out, "topology_history.png")
    fig.savefig(history_path, dpi=150)
    plt.close(fig)

    tile = 4
    rho_np = np.asarray(rho_hat)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rho_np, origin="lower", cmap="binary", vmin=0, vmax=1)
    axes[0].set_title("Projected density")
    axes[1].imshow(np.tile(rho_np, (tile, tile)), origin="lower", cmap="binary",
                   vmin=0, vmax=1)
    axes[1].set_title(f"{tile}x{tile} tiled")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    density_path = os.path.join(args.out, "topology_density.png")
    fig.savefig(density_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {bz_path}, {history_path} and {density_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
