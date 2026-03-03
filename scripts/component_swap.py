#!/usr/bin/env python3
"""
Memory-safe checkpoint merge script for OpenPI / Flax NNX.

Loads ckpt1 fully (base model), restores only selected subtrees
(e.g., action-head, vision, language) from ckpt2 via Orbax partial
restore, overwrites those subtrees into ckpt1, and saves the
merged checkpoint to a new directory.
"""

import dataclasses
import gc
import logging
import platform
from typing import Any, Dict

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


# ── Weight loading helpers ──────────────────────────────────────

def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _load_weights_and_validate_partial(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates a partial checkpoint, tolerating missing/extra keys."""
    loaded_params = loader.load(params_shape)

    exp_flat = traverse_util.flatten_dict(params_shape, keep_empty_nodes=True)
    got_flat = traverse_util.flatten_dict(loaded_params, keep_empty_nodes=True)

    common_keys = set(exp_flat.keys()) & set(got_flat.keys())
    missing_keys = set(exp_flat.keys()) - set(got_flat.keys())
    extra_keys = set(got_flat.keys()) - set(exp_flat.keys())

    if missing_keys:
        logging.info(f"Checkpoint is missing {len(missing_keys)} parameter(s). Examples: {list(missing_keys)[:5]}")
    if extra_keys:
        logging.info(f"Checkpoint has {len(extra_keys)} unexpected parameter(s). Examples: {list(extra_keys)[:5]}")

    exp_common = traverse_util.unflatten_dict({k: exp_flat[k] for k in common_keys})
    got_common = traverse_util.unflatten_dict({k: got_flat[k] for k in common_keys})

    at.check_pytree_equality(expected=exp_common, got=got_common, check_shapes=True, check_dtypes=True)

    cleaned = {
        k: v for k, v in traverse_util.flatten_dict(got_common).items()
        if not isinstance(v, jax.ShapeDtypeStruct)
    }
    return traverse_util.unflatten_dict(cleaned)


# ── Partial restore + overwrite ─────────────────────────────────

def _build_regex_subset_template_params(
    train_state_template: training_utils.TrainState,
    regex: str,
) -> Dict:
    """Build a pure-dict template containing only params matching regex."""
    subset_filter = nnx_utils.PathRegex(regex)
    subset_state = train_state_template.params.filter(subset_filter)
    return subset_state.to_pure_dict()


def _restore_params_by_regex_only(
    ckpt_dir: epath.Path,
    step: int,
    train_state_template: training_utils.TrainState,
    mesh: jax.sharding.Mesh,
    *,
    regex: str,
) -> Dict:
    """Restore only params matching `regex` from ckpt_dir/<step>/params."""
    params_path = ckpt_dir / str(step) / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"params checkpoint dir not found: {params_path}")

    subset_template = _build_regex_subset_template_params(train_state_template, regex)

    # Inspect on-disk structure and filter to common keys
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        metadata = checkpointer.metadata(params_path)
    except Exception as e:
        logging.warning(f"Could not read checkpoint metadata from {params_path}: {e}")
        metadata = None

    if metadata is not None:
        flat_template = traverse_util.flatten_dict(subset_template, sep="/")
        try:
            flat_metadata = traverse_util.flatten_dict(metadata, sep="/")
            common_keys = set(flat_template.keys()) & set(flat_metadata.keys())
            missing_in_ckpt = set(flat_template.keys()) - set(flat_metadata.keys())

            if missing_in_ckpt:
                logging.info(
                    f"[partial restore] {len(missing_in_ckpt)} requested keys missing in checkpoint. "
                    f"Examples: {list(missing_in_ckpt)[:5]}"
                )

            filtered_flat = {k: flat_template[k] for k in common_keys}
            subset_template = traverse_util.unflatten_dict(filtered_flat, sep="/")
        except Exception as e:
            logging.warning(f"Could not filter template by metadata keys: {e}. Using full template.")

    if not subset_template:
        logging.warning(f"[partial restore] No matching keys for regex '{regex}'. Returning empty dict.")
        return {}

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def _mk_restore_args(x):
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return ocp.ArrayRestoreArgs(sharding=replicated_sharding)
        return None

    restore_args = jax.tree.map(_mk_restore_args, subset_template)

    logging.info(f"[partial restore] regex={regex}, from: {params_path}")

    return checkpointer.restore(
        params_path,
        item=subset_template,
        restore_args=restore_args,
    )


def _overwrite_params_subset_in_state(
    state_base: training_utils.TrainState,
    subset_params_pure_dict: Dict,
) -> training_utils.TrainState:
    """Overwrite only the keys in subset_params_pure_dict into state_base.params."""
    model = nnx.merge(state_base.model_def, state_base.params)
    graphdef, nnx_state = nnx.split(model)
    nnx_state.replace_by_pure_dict(subset_params_pure_dict)
    new_model = nnx.merge(graphdef, nnx_state)
    new_params = nnx.state(new_model)
    return dataclasses.replace(state_base, params=new_params)


def _free_memory():
    """Best-effort memory cleanup between restores/merges."""
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass


# ── Merge + Save ────────────────────────────────────────────────

def run_merge_and_save(config, train_state_template, mesh, data_sharding):
    """
    Memory-safe merge:
      1. Restore ckpt1 fully
      2. Restore only selected subtrees from ckpt2
      3. Overwrite into ckpt1
      4. Save merged checkpoint
    """
    ckpt1_dir = epath.Path(config.ckpt1_path)
    ckpt1_step = config.merge_base_step

    ckpt2_dir = epath.Path(config.ckpt2_path)
    ckpt2_step = config.merge_action_step

    merge_vision = bool(config.merge_vision)
    merge_language = bool(config.merge_language)
    merge_action = bool(config.merge_action)
    merge_projs = bool(config.merge_extra_projections)

    output_dir_name = (
        f"{ckpt1_dir.name}_base{ckpt1_step}_action{ckpt2_step}_merged_"
        f"v{int(merge_vision)}l{int(merge_language)}a{int(merge_action)}"
    )
    output_dir = ckpt1_dir.parent / output_dir_name

    logging.info("=== STARTING MERGE OPERATION ===")
    logging.info(f"Base ckpt:   {ckpt1_dir} @ step {ckpt1_step}")
    logging.info(f"Action ckpt: {ckpt2_dir} @ step {ckpt2_step}")
    logging.info(f"Merge vision={merge_vision}, lang={merge_language}, action={merge_action}, projs={merge_projs}")
    logging.info(f"Output dir:  {output_dir}")

    # 1) Restore ckpt1 fully
    logging.info("[1/4] Restoring base checkpoint...")
    mgr_base, _ = _checkpoints.initialize_checkpoint_dir(
        ckpt1_dir, keep_period=config.keep_period, overwrite=False, resume=True,
    )
    state_base, _, _ = _checkpoints.restore_state(
        mgr_base, train_state_template, step=ckpt1_step,
        data_loader=None, restore_task_idx=False, restore_replay_buffer=False,
    )
    _free_memory()

    # 2) Partial restore from ckpt2
    logging.info("[2/4] Partially restoring selected subtrees from ckpt2...")

    subtrees = []
    if merge_vision:
        subtrees.append(("vision", r".*img.*"))
    if merge_language:
        subtrees.append(("language", r".*llm.*"))
    if merge_action:
        subtrees.append(("action", r".*llm.*_1.*"))
    if merge_projs:
        subtrees.append(("projections", r".*(action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out|state_proj).*"))

    for name, regex in subtrees:
        logging.info(f"  Restoring {name} subtree...")
        subset_params = _restore_params_by_regex_only(
            ckpt_dir=ckpt2_dir, step=ckpt2_step,
            train_state_template=train_state_template, mesh=mesh, regex=regex,
        )
        state_base = _overwrite_params_subset_in_state(state_base, subset_params)
        del subset_params
        _free_memory()

    logging.info("[merge] Finished overwriting partial params from ckpt2.")

    # 3) Save merged checkpoint
    logging.info("[3/4] Saving merged state...")
    mgr_save, _ = _checkpoints.initialize_checkpoint_dir(
        output_dir, keep_period=config.keep_period,
        overwrite=config.overwrite, resume=config.resume,
    )

    n_tasks = config.n_tasks
    task_loaders = [
        _data_loader.create_data_loader_sequential(
            config, task_id=tid, shuffle=True,
            sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)),
            **({
                "cust_batch_size": int(config.batch_size * 2)
            } if tid == 0 and config.double_batch_start else {}),
        )
        for tid in range(n_tasks)
    ]

    save_step = ckpt1_step
    _checkpoints.save_state(
        mgr_save, state_base, data_loader=task_loaders[0], step=save_step,
        task_loaders=task_loaders, replay_loader=None,
        task_idx=ckpt2_step // config.steps_per_task, buffer_lst=None,
    )

    mgr_save.wait_until_finished()
    logging.info(f"[4/4] Done! Saved merged checkpoint to {output_dir}/{save_step}")


# ── Train state init ────────────────────────────────────────────

@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    with jax.default_device(jax.devices("cpu")[0]):
        train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    if config.train_from_scratch is None:
        partial_params = _load_weights_and_validate_partial(
            config.weight_loader, train_state_shape.params.to_pure_dict()
        )
    elif config.train_from_scratch == "from_scratch":
        partial_params = None
    else:
        partial_params = None

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init, donate_argnums=(1,),
        in_shardings=replicated_sharding, out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# ── Main ────────────────────────────────────────────────────────

def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    init_wandb(config, resuming=False, enabled=config.wandb_enabled)

    train_state, _ = init_train_state(config, init_rng, mesh, resume=False)
    run_merge_and_save(config, train_state, mesh, data_sharding)


if __name__ == "__main__":
    main(_config.cli())
