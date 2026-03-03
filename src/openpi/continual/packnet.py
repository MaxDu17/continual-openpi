# openpi/training/packnet.py

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import traverse_util
import flax.nnx as nnx
import optax

from openpi.shared import nnx_utils
import openpi.shared.array_typing as at

IMG_PREFIX = "PaliGemma/img/"
LLM_PREFIX = "PaliGemma/llm/"

# -----------------------------
# state
# -----------------------------

@dataclasses.dataclass
class PackNetState:
    """Holds per‑weight ownership masks and the current task id.

    masks: pytree with same structure as params. Each kernel leaf is an int32 array:
      0          -> free / unassigned
      task_idx+1 -> weight owned by that task
    For non‑kernel leaves, we store None (to avoid unnecessary allocations).
    """
    current_task: int
    masks: Any  # PyTree (same keys as params). kernel leaves: int32 arrays, non‑kernel leaves: None


# -----------------------------
# Helpers
# -----------------------------

def _flatten(tree: Any) -> dict[tuple[str, ...], Any]:
    """Flatten to a dict with tuple keys; works with nnx.State or nested dicts."""
    if hasattr(tree, "to_pure_dict"):  # nnx.State
        pure = tree.to_pure_dict()
    else:
        pure = tree
    return traverse_util.flatten_dict(pure, keep_empty_nodes=True)

def _unflatten_like(flat: dict[tuple[str, ...], Any]) -> Any:
    return traverse_util.unflatten_dict(flat)

def _select_task_owned(weights: jnp.ndarray, mask: jnp.ndarray, task_plus_1: int) -> jnp.ndarray:
    """Return 1/0 mask (same shape) where weights belong to (current_task+1)."""
    return (mask == task_plus_1).astype(jnp.int32)

def _kth_smallest_abs(x: jnp.ndarray, k: int) -> jnp.ndarray:
    """0-based kth smallest of |x|, JIT friendly. If k < 0 or x.size == 0, returns +inf."""
    size = x.size
    def _body():
        # jnp.partition works on flattened arrays.
        flat = jnp.abs(x).reshape(-1)
        # Clamp k to valid range to avoid XLA errors.
        kk = jnp.clip(k, 0, jnp.maximum(size - 1, 0))
        part = jnp.partition(flat, kk)
        return part[kk]
    return jax.lax.cond((size > 0) & (k >= 0), _body, lambda: jnp.array(jnp.inf, x.dtype))


def _path_str(path: tuple[str, ...]) -> str:
    return "/".join(path)

def should_mask_param_for_your_model(path: tuple[str, ...], x: jnp.ndarray) -> bool:
    if not isinstance(x, jnp.ndarray):
        return False
    name = _path_str(path)
    leaf = path[-1] if path else ""

    # 1) Vision branch: train everything except bias/norm
    if name.startswith(IMG_PREFIX):
        return leaf not in ("bias", "scale")

    # 2) LLM branch: LoRA only
    if name.startswith(LLM_PREFIX):
        return ("lora" in name) and (leaf != "bias")

    # 3) Policy heads: include kernels, freeze biases
    if name.startswith(("action_", "state_proj")):
        return leaf != "bias"   # kernels, mlp weights, etc.

    return False


# -----------------------------
# API
# -----------------------------

trainable_filter = nnx.All(
    nnx.Param,
    lambda path, node: should_mask_param_for_your_model(path, node),
)

freeze_filter = nnx.All(
    nnx.Param, 
    lambda path, node: not should_mask_param_for_your_model(path, node))


def init_masks(params, cfg) -> any:
    fparams = traverse_util.flatten_dict(params.to_pure_dict(), keep_empty_nodes=True)
    flat_masks = {}
    for path, p in fparams.items():
        if should_mask_param_for_your_model(path, p):
            flat_masks[path] = jnp.zeros(p.shape, dtype=jnp.int32)
        else:
            flat_masks[path] = None  # no per‑weight mask (frozen by PackNet)
    return traverse_util.unflatten_dict(flat_masks)


def start_task(pstate: PackNetState, params: at.Params, cfg: PackNetConfig) -> PackNetState:
    """Assign all currently free (0) kernel weights to the new task (task_id + 1)."""
    task_plus_1 = pstate.current_task + 1
    fmasks = _flatten(pstate.masks)

    new_flat = {}
    for path, m in fmasks.items():
        if m is None:
            new_flat[path] = None
        else:
            # Assign free slots (0) to task_plus_1
            new_flat[path] = jnp.where(m == 0, jnp.int32(task_plus_1), m).astype(jnp.int32)
    new_masks = _unflatten_like(new_flat)
    return dataclasses.replace(pstate, masks=new_masks)


def apply_update_mask(
    updates: Any, masks: Any, current_task: int, cfg: PackNetConfig
) -> Any:
    """Zero out updates that do not belong to the current task; freeze biases/norms."""
    fupd = _flatten(updates)
    fmasks = _flatten(masks)

    task_plus_1 = current_task + 1
    new_fupd = {}

    for path, u in fupd.items():
        if not isinstance(u, jnp.ndarray):  # optax can carry scalars in pytrees; pass through
            new_fupd[path] = u
            continue

        m = fmasks.get(path, None)
        if m is None:
            # Non‑kernel (bias/norm/embeddings) → freeze: zero updates.
            new_fupd[path] = jnp.zeros_like(u)
        else:
            # Per‑weight masking: keep only owner == current_task+1
            keep = (m == task_plus_1).astype(u.dtype)
            # Broadcastable safety check: shapes are identical because masks were built from params.
            new_fupd[path] = u * keep
    return _unflatten_like(new_fupd)


def apply_update_mask_state(
    updates: nnx.State, masks: nnx.State | dict, current_task: int, cfg: PackNetConfig
) -> nnx.State:
    """Return masked updates with the SAME container type/structure as `updates` (nnx.State)."""
    # Flatten both trees to dicts
    up_flat = traverse_util.flatten_dict(updates.to_pure_dict(), keep_empty_nodes=True)
    ms_flat = (traverse_util.flatten_dict(masks.to_pure_dict(), keep_empty_nodes=True)
               if isinstance(masks, nnx.State) else
               traverse_util.flatten_dict(masks, keep_empty_nodes=True))

    t1 = current_task + 1
    out_flat = {}
    for path, u in up_flat.items():
        m = ms_flat.get(path, None)
        if m is None:
            out_flat[path] = jnp.zeros_like(u)  # freeze
        else:
            keep = (m == t1).astype(u.dtype)
            out_flat[path] = u * keep

    # Unflatten to a nested dict with the same keys as `updates`
    out_nested = traverse_util.unflatten_dict(out_flat)

    # Clone the updates state or reuse it; then replace leaves by dict.
    masked_updates = updates  # safe to reuse; this object is ephemeral in the step
    masked_updates.replace_by_pure_dict(out_nested)
    return masked_updates

def prune_once(
    params: at.Params,
    pstate: PackNetState,
    cfg: PackNetConfig,
    *,
    host_log: Callable[[str], None] | None = print,
) -> tuple[at.Params, PackNetState]:
    """Magnitude prune the weights *owned by the current task*.

    Returns:
      new_params: params with pruned weights zeroed.
      new_pstate: updated masks where pruned indices are set to 0 (free).
    """
    task_plus_1 = pstate.current_task + 1
    if not (0.0 <= cfg.prune_perc < 1.0):
        raise ValueError(f"prune_perc must be in [0, 1). Got {cfg.prune_perc}")

    fparams = _flatten(params)
    fmasks  = _flatten(pstate.masks)

    new_fparams: dict[tuple[str, ...], Any] = {}
    new_fmasks: dict[tuple[str, ...], Any] = {}

    for path, w in fparams.items():
        m = fmasks[path]
        if m is None or not isinstance(w, jnp.ndarray):
            # Non‑kernel: carry through unchanged.
            new_fparams[path] = w
            new_fmasks[path]  = m
            continue

        owned = (m == task_plus_1)
        num_owned = owned.sum()

        def _no_owned():
            return w, m

        def _prune_owned():
            n = num_owned 
            kth_zero_based = jnp.maximum(0, jnp.int32(jnp.round(cfg.prune_perc * n)) - 1)

            # mask-with-∞ trick (no boolean gather)
            flat_abs = jnp.abs(w).reshape(-1)
            flat_msk = owned.reshape(-1)
            inf_val  = jnp.asarray(jnp.inf, flat_abs.dtype)   # keep dtype (bf16/f32)
            masked   = jnp.where(flat_msk, flat_abs, inf_val) # shape: [N]

            # Instead of jnp.partition(masked, kk)[kk] (k must be static),
            # sort then use a dynamic index (jit-friendly).
            kk = jnp.clip(kth_zero_based, 0, masked.size - 1)
            sorted_vals = jnp.sort(masked)  # ascending, O(N log N) but called once per task
            cutoff = jax.lax.dynamic_index_in_dim(sorted_vals, kk, axis=0, keepdims=False)

            remove = (jnp.abs(w) <= cutoff) & owned
            m_new  = jnp.where(remove, jnp.int32(0), m).astype(jnp.int32)
            w_new  = jnp.where(remove, jnp.zeros_like(w), w)
            return w_new, m_new

        # def _prune_owned():
        #     # Compute cutoff rank among abs(weights[owned]).
        #     n = num_owned
        #     # Round like torch: round(p * n) then convert to 0‑based kth
        #     # kth_zero_based = jnp.maximum(0, jnp.int32(jnp.round(cfg.prune_perc * n)) - 1)
        #     # cutoff = _kth_smallest_abs(w[owned], kth_zero_based)
        #     # remove = (jnp.abs(w) <= cutoff) & owned  # to be freed
            
        #     ################
            
        #     # n = num_owned (already defined)
        #     kth_zero_based = jnp.maximum(0, jnp.int32(jnp.round(cfg.prune_perc * n)) - 1)

        #     # mask-with-∞ trick: kth among owned == kth among all when others are +inf
        #     flat_abs = jnp.abs(w).reshape(-1)
        #     flat_msk = owned.reshape(-1)
        #     inf_val  = jnp.array(jnp.inf, flat_abs.dtype)  # preserves bfloat16/float32
        #     masked   = jnp.where(flat_msk, flat_abs, inf_val)

        #     # safe index (still derived from n, but clip to array size for XLA safety)
        #     kk = jnp.clip(kth_zero_based, 0, masked.size - 1)
        #     cutoff = jnp.partition(masked, kk)[kk]

        #     # now build remove mask on the original shape
        #     remove = (jnp.abs(w) <= cutoff) & owned
            
        #     ################
            
        #     # Update mask: set removed positions to 0, keep others unchanged.
        #     m_new = jnp.where(remove, jnp.int32(0), m).astype(jnp.int32)
        #     # Zero out pruned weights in params for stability (like torch code).
        #     w_new = jnp.where(remove, jnp.zeros_like(w), w)

        #     if cfg.log_pruning and host_log is not None:
        #         # Host prints only; avoid massive logging in pjit contexts.
        #         total_in_layer = w.size
        #         pruned_cnt = remove.sum()
        #         pct = (pruned_cnt.astype(jnp.float32) / jnp.maximum(n.astype(jnp.float32), 1.0)) * 100.0
        #         host_log(
        #             f"PackNet prune { '/'.join(path) }: pruned {int(pruned_cnt)} / {int(n)} "
        #             f"({float(pct):.2f}%) | total elems: {int(total_in_layer)}"
        #         )
        #     return w_new, m_new

        w_out, m_out = jax.lax.cond(num_owned == 0, _no_owned, _prune_owned)
        new_fparams[path] = w_out
        new_fmasks[path]  = m_out

    return _unflatten_like(new_fparams), dataclasses.replace(pstate, masks=_unflatten_like(new_fmasks))


def prune_once_arrays(params, masks_dict, current_task: int, cfg: PackNetConfig):
    """JIT‑friendly wrapper: takes/returns only array pytrees (dicts)."""
    # call the existing prune_once, but DO NOT print inside jit
    new_params_dict, new_pstate = prune_once(
        params,
        PackNetState(current_task=current_task, masks=masks_dict),
        cfg,
        host_log=None,
    )
    return new_params_dict, new_pstate.masks 


def eval_param_masking(params: at.Params, masks: Any, upto_task_id: int, cfg: PackNetConfig) -> at.Params:
    """Create params view for evaluation on a specific task:
       - zero weights with mask == 0 (free)
       - zero weights with mask > upto_task_id+1 (future tasks)
    """
    fparams = _flatten(params)
    fmasks  = _flatten(masks)
    thresh  = upto_task_id + 1

    out_flat = {}
    for path, w in fparams.items():
        m = fmasks[path]
        if m is None or not isinstance(w, jnp.ndarray):
            out_flat[path] = w
        else:
            keep = (m > 0) & (m <= thresh)
            out_flat[path] = jnp.where(keep, w, jnp.zeros_like(w))
    return _unflatten_like(out_flat)
