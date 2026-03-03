"""
Elastic Weight Consolidation – *online* variant
==============================================

This module provides an **online EWC** helper that is fully compatible with
*openpi*'s Flax/NNX training loop.  Compared to the canonical implementation
(which stores a separate Fisher diagonal **Fₜ** and parameter snapshot
**θₜ\*** for every past task), the *online* formulation (Schwarz *et al.*, 2018)
maintains **one** running Fisher approximation and **one** parameter snapshot:

\[F ← γ·F + F\_new\] and \[θ* ← θ\_new\]

This reduces memory usage from \(O(T·|θ|)\) to \(O(|θ|)\) at the cost of an
exponential decay controlled by **γ ∈ [0,1]**.

---
High‑level usage
----------------
```python
from openpi.training.algorithms.ewc import OnlineEWC

# create once
ewc = OnlineEWC(lambda_=0.4, gamma=0.95)

for task_id, loader in enumerate(task_loaders):
    train_state = run_single_task(train_state, loader, ewc)
    ewc.compute_fisher(
        model_def=train_state.model_def,
        params=train_state.params,
        data_loader=loader,
        rng=train_rng,
        num_batches=config.ewc_n_fisher_batches,
    )
```
Inside `run_single_task` simply add the penalty:
```python
loss = task_loss + ewc.penalty(state.params)
```
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Iterable, Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax import tree_util

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _tree_sum(tree):
    """Sum all leaves (scalars or tensors) of a PyTree into a single scalar."""
    return tree_util.tree_reduce(lambda a, b: a + b, tree, 0.0)


def _param_values(p):  # noqa: D401 – simple helper
    """Return the *array* held by an NNX `Param` – or the object itself."""
    return p.value if hasattr(p, "value") else p


# -----------------------------------------------------------------------------
# Online EWC implementation
# -----------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
class EWC:  # noqa: D401 – keep class name concise
    """Memory‑efficient **online Elastic Weight Consolidation** helper.

    Parameters
    ----------
    lambda_ : float
        Regularisation strength λ.
    gamma : float, default = 1.0
        Decay applied to the running Fisher diagonal after each task.  Set
        **γ < 1** to gradually forget very old tasks; **γ = 1** reproduces
        original EWC but with *O(1)* memory (recent tasks dominate numerically
        after many updates).
    """

    # ------------------------------------------------------------------
    # Construction & state
    # ------------------------------------------------------------------

    def __init__(self, *, lambda_: float, gamma: float = 1.0, 
                 fisher_diag = None, prev_params = None) -> None:  # noqa: D401
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be within [0, 1]")
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)

        self._fisher_diag = fisher_diag
        self._prev_params = prev_params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tree_flatten(self):
        children = (self._fisher_diag, self._prev_params)
        aux_data = (self.lambda_, self.gamma)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        fisher_diag, prev_params = children
        lambda_, gamma = aux_data
        return cls(lambda_=lambda_, gamma=gamma,
                   fisher_diag=fisher_diag, prev_params=prev_params)


    def penalty(self, params) -> jnp.ndarray:
        """Return λ / 2 · Σᵢ Fᵢ (θᵢ − θ*ᵢ)².

        If `compute_fisher` has **not** been called yet, this returns 0.
        """
        
        if self._fisher_diag is None or self._prev_params is None:
            return jnp.asarray(0.0, dtype=jnp.float32)

        penalty_val = _tree_sum(
            tree_util.tree_map(
                lambda f, θ_prev, θ_now: jnp.sum(f * jnp.square(_param_values(θ_now) - _param_values(θ_prev))),  # type: ignore[arg-type]
                self._fisher_diag,
                self._prev_params,
                params,
            )
        )
        return 0.5 * self.lambda_ * penalty_val

    # ------------------------------------------------------------------
    # After each task finishes
    # ------------------------------------------------------------------

    def compute_fisher(
        self,
        *,
        model_def: nnx.GraphDef,
        params: nnx.State,
        data_loader: Iterable,  # yields (Observation, Actions)
        rng: jax.Array,
        num_batches: Optional[int] = 100,
    ) -> None:
        
        mdl = nnx.merge(model_def, params)
        mdl.eval()

        @jax.jit
        def per_batch_fisher(p, key, obs, act):
            nnx.update(mdl, p)
            loss = jnp.mean(mdl.compute_loss(key, obs, act, train=False))
            sg_grads = jax.grad(lambda pp: loss)(p)
            return tree_util.tree_map(lambda g: g * g, sg_grads)

        fisher_new = tree_util.tree_map(lambda x: jnp.zeros_like(_param_values(x)), params)
        
        dl_iter = iter(data_loader)
        n_batches = 0
        for b_idx, (obs, act) in enumerate(dl_iter):
            print(f"At batch idx {b_idx}")
            if num_batches > 0 and b_idx >= num_batches:
                break
            rng, subkey = jax.random.split(rng)
            fisher_new = tree_util.tree_map(
                jnp.add,
                fisher_new,
                per_batch_fisher(params, subkey, obs, act),
            )
            n_batches += 1

        if n_batches == 0:
            raise ValueError("`compute_fisher` consumed zero batches; check loader.")

        fisher_new = tree_util.tree_map(lambda x: x / n_batches, fisher_new)

        # --------------------------------------------------------------
        # Online update: F ← γF + F_new, θ* ← θ
        # --------------------------------------------------------------

        if self._fisher_diag is None:
            self._fisher_diag = fisher_new
        else:
            self._fisher_diag = tree_util.tree_map(
                lambda old, new: self.gamma * old + new,
                self._fisher_diag,
                fisher_new,
            )
        self._prev_params = params

        def _to_bf16(tree):
            return jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), tree)

        self._fisher_diag = _to_bf16(self._fisher_diag)
        self._prev_params = _to_bf16(copy.deepcopy(self._prev_params))

    # ------------------------------------------------------------------
    # Diagnostics / pretty‑printing
    # ------------------------------------------------------------------

    def __repr__(self):  # noqa: D401
        has_fish = self._fisher_diag is not None
        return (
            "OnlineEWC("f"lambda_={self.lambda_}, gamma={self.gamma}, "
            f"initialised={'yes' if has_fish else 'no'})"
        )
