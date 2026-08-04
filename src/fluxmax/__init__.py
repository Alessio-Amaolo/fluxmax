"""Near-field radiative heat transfer via RCWA trace formula.

Runtime value checks
--------------------
The numeric path carries ``chex`` value assertions.
Value assertions cannot be evaluated while tracing unless the *outermost* jitted
function is wrapped in :func:`chex.chexify`, so they are disabled here at import and
cost nothing by default: with ``chex.disable_asserts()`` in force they are a no-op
under eager execution, ``jax.jit``, ``jax.vmap`` and ``jax.grad`` alike.

To turn them on::

    import chex
    chex.enable_asserts()

    @chex.chexify          # must sit on top of every jit/vmap/pmap
    @jax.jit
    def loss(params):
        ...                # calls into fluxmax
    value = loss(params)
    loss.wait_checks()     # or chex.block_until_chexify_assertions_complete()
"""

import chex

from . import materials, parallelism

chex.disable_asserts()

__all__ = ["materials", "parallelism"]
