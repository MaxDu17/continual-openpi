import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm_loggable.auto as tqdm
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

from openpi.training.data_loader import create_deterministic_buffer, create_torch_data_loader_replay_buffer, MixedDataLoader


# Monkey-patch to fix 'List' feature type error in old datasets
try:
    import datasets.features.features as features

    _OLD_GENERATE_FROM_DICT = features.generate_from_dict

    def _new_generate_from_dict(obj):
        if isinstance(obj, dict) and obj.get("_type") == "List":
            obj["_type"] = "Sequence"
        return _OLD_GENERATE_FROM_DICT(obj)

    features.generate_from_dict = _new_generate_from_dict
    print("successfully patched list error")
except (ImportError, AttributeError):
    # If datasets or the function doesn't exist, do nothing.
    pass
# End of monkey-patch


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
    logger.handlers[0].setFormatter(formatter)


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


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _load_weights_and_validate_partial(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads a checkpoint subset that matches `params_shape`.
    - Validates shape/dtype for overlapping keys.
    - Ignores missing keys (keeps model's init values, e.g., new params like `action_query`).
    - Drops unexpected keys present in the checkpoint.
    """
    loaded_params = loader.load(params_shape)

    # Flatten for set ops; keys are tuples of path components.
    exp_flat = traverse_util.flatten_dict(params_shape, keep_empty_nodes=True)
    got_flat = traverse_util.flatten_dict(loaded_params, keep_empty_nodes=True)

    exp_keys = set(exp_flat.keys())
    got_keys = set(got_flat.keys())

    common_keys = exp_keys & got_keys
    missing_keys = exp_keys - got_keys
    extra_keys = got_keys - exp_keys

    if missing_keys:
        logging.info(
            f"Checkpoint is missing {len(missing_keys)} parameter(s); using model init for them. "
            f"Examples: {list(missing_keys)[:5]}",
        )
    if extra_keys:
        logging.info(
            f"Checkpoint has {len(extra_keys)} unexpected parameter(s); dropping them. "
            f"Examples: {list(extra_keys)[:5]}",
        )

    # Build pruned trees with only overlapping keys.
    exp_common = traverse_util.unflatten_dict({k: exp_flat[k] for k in common_keys})
    got_common = traverse_util.unflatten_dict({k: got_flat[k] for k in common_keys})

    # Validate shapes/dtypes on the intersection only.
    at.check_pytree_equality(
        expected=exp_common,
        got=got_common,
        check_shapes=True,
        check_dtypes=True,
    )

    # Return only the loaded params (intersection), stripping ShapeDtypeStruct placeholders.
    cleaned = {
        k: v
        for k, v in traverse_util.flatten_dict(got_common).items()
        if not isinstance(v, jax.ShapeDtypeStruct)
    }
    return traverse_util.unflatten_dict(cleaned)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # Initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
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
            config.weight_loader,
            train_state_shape.params.to_pure_dict(),
        )
    elif config.train_from_scratch == "from_scratch":
        partial_params = None
    elif config.train_from_scratch == "from_paligemma":
        partial_params = _weight_loaders.HuggingFaceWeightLoader(
            repo_id="google/paligemma-3b-pt-224",
            include_llm=True,
        ).load(train_state_shape.params.to_pure_dict())
    elif config.train_from_scratch == "from_paligemma_vision":
        partial_params = _weight_loaders.HuggingFaceWeightLoader(
            repo_id="google/paligemma-3b-pt-224",
            include_llm=False,
        ).load(train_state_shape.params.to_pure_dict())
    else:
        raise ValueError(f"Unknown train_from_scratch value: {config.train_from_scratch}")

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    info = {"loss": loss}
    return new_state, info


def make_single_task_loop(
    config, mesh, train_rng, train_state, train_state_sharding,
    data_sharding, replicated_sharding, data_loader, task_loaders,
    task_idx, checkpoint_manager, replay_loader=None, buffer_lst=None,
):
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    global_start = int(jax.device_get(train_state.step))

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=global_start)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    global_end = global_start + config.steps_per_task

    pbar = tqdm.tqdm(
        range(global_start, global_end),
        initial=global_start,
        total=global_end,
        dynamic_ncols=True,
        desc=f"T{task_idx}",
    )

    infos = []
    for global_step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)

        jax.block_until_ready(train_state)
        infos.append(info)

        if global_step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {global_step}: {info_str}")
            wandb.log(reduced_info, step=global_step)
            infos = []

        if global_step % config.save_interval == 0 and global_step > 0:
            _checkpoints.save_state(
                checkpoint_manager, train_state, data_loader, global_step,
                task_loaders, replay_loader=replay_loader,
                task_idx=task_idx, buffer_lst=buffer_lst,
            )

        batch = next(data_iter)

    _checkpoints.save_state(
        checkpoint_manager, train_state, data_loader, global_step,
        task_loaders, replay_loader=replay_loader,
        task_idx=task_idx, buffer_lst=buffer_lst,
    )

    return train_state


def _latest_9999_step(checkpoint_manager: ocp.CheckpointManager) -> int | None:
    """Find the latest checkpoint step ending with '9999' (end-of-task boundary)."""
    checkpoint_manager.reload()
    all_steps = checkpoint_manager.all_steps(read=True)
    candidate_steps = [s for s in all_steps if str(s).endswith("9999")]
    return max(candidate_steps) if candidate_steps else None


def _prune_future_checkpoints(checkpoint_manager: ocp.CheckpointManager, keep_step: int):
    """Delete any checkpoints beyond keep_step."""
    checkpoint_manager.reload()
    for s in checkpoint_manager.all_steps(read=True):
        if s > keep_step:
            logging.info(f"Deleting future checkpoint at step {s}")
            checkpoint_manager.delete(s)


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
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    n_tasks = config.n_tasks

    logging.info("Creating per-task dataloaders")
    # task loader is what brings the sequence task up 
    task_loaders = [
        _data_loader.create_data_loader_sequential(
            config,
            task_id=tid,
            shuffle=True,
            sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)),
            **({
                "cust_batch_size": int(config.batch_size * 2)
            } if tid == 0 and config.double_batch_start else {}),
        )
        for tid in range(n_tasks)
    ]

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # Log param counts.
    def _count_params_state(state_obj) -> int:
        pd = state_obj.to_pure_dict()
        leaves = jax.tree_util.tree_leaves(pd)
        total = 0
        for leaf in leaves:
            if isinstance(leaf, nnx.Param):
                total += int(leaf.value.size)
            elif hasattr(leaf, "size"):
                total += int(leaf.size)
        return total

    trainable_state = train_state.params.filter(config.trainable_filter)
    frozen_state = train_state.params.filter(config.freeze_filter)
    total_params = _count_params_state(train_state.params)
    trainable_params = _count_params_state(trainable_state)
    frozen_params = _count_params_state(frozen_state)
    logging.info(f"[params] total={total_params:,} trainable={trainable_params:,} frozen={frozen_params:,}")

    prev_task_idx = -1

    # Task order (for ablation studies).
    order = None
    if config.cl_order == "243":
        order = [0, 1, 2, 4, 3]
    elif config.cl_order == "324":
        order = [0, 1, 3, 2, 4]
    elif config.cl_order == "342":
        order = [0, 1, 3, 4, 2]
    elif config.cl_order == "423":
        order = [0, 1, 4, 2, 3]

    logging.info(f"Using task order: {order}")

    # to dynamically assign, run create_deterministic_buffer at the start of every task
    # before doing this, make your data filter! 
    dataset_lst, buffer_lst = create_deterministic_buffer(
        config,
        n_tasks=n_tasks,
        shuffle=True,
        sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)),
        task_order=order, # this is how you determine the task order of loading 
    )
    # 
    # ^^^ preset_buffers is how you select the subsets 
    # ^^^^ this function creates the dataset torch object and the sampling buffer for each tasks, 
    # created in the order of loading 


    if resuming:
        if config.cl == "er":
            step_to_restore = _latest_9999_step(checkpoint_manager)
            logging.info(f"Restoring from step: {step_to_restore}")

            train_state, prev_task_idx, saved_buffer_lst = _checkpoints.restore_state(
                checkpoint_manager, train_state, data_loader=None,
                restore_task_idx=True, restore_replay_buffer=True, step=step_to_restore,
            )
            _prune_future_checkpoints(checkpoint_manager, step_to_restore)

            dataset_lst, buffer_lst = create_deterministic_buffer(
                config,
                n_tasks=n_tasks,
                shuffle=True,
                sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)),
                task_order=order,
                preset_buffers=saved_buffer_lst,
            )

            dataset_lst_final = (
                dataset_lst[:prev_task_idx + 1]
                if not config.no_resampling
                else dataset_lst[:prev_task_idx + 2]
            )

            replay_loader = create_torch_data_loader_replay_buffer(
                dataset_lst_final,
                task_loaders[prev_task_idx + 1].data_config(),
                model_config=config.model,
                action_horizon=config.model.action_horizon,
                batch_size=config.batch_size,
                sharding=data_sharding,
                shuffle=True,
                num_batches=None,
                num_workers=config.num_workers,
                seed=config.seed,
                skip_norm_stats=False,
            )
        else:
            train_state, prev_task_idx, _ = _checkpoints.restore_state(
                checkpoint_manager, train_state, data_loader=None, restore_task_idx=True,
            )
            replay_loader = None

        assert prev_task_idx + 1 >= 0
        last_step = checkpoint_manager.latest_step()
        logging.info(f"Resuming from checkpoint step {last_step}, task {prev_task_idx}")
    else:
        replay_loader = None

    # --- Sequential training loop ---
    def _run_task(task_idx):
        # this is the task to run on 
        nonlocal train_state, replay_loader

        loader = task_loaders[task_idx]
        logging.info(f"=== Learning task {task_idx} / {n_tasks - 1} ===")

        if config.cl == "er":
            # this is where the ER is implemented...
            if config.no_resampling: # this doesn't do balanced batches; it just includes the current task in the sampling 
                mixed_loader = replay_loader if task_idx > 0 else loader
            else:
                # this concatenates the training batch and ER batch 
                mixed_loader = MixedDataLoader(loader.data_config(), loader, replay_loader) if task_idx > 0 else loader
        else:
            mixed_loader = loader

        # this is the actual training loop 
        train_state = make_single_task_loop(
            config, mesh, train_rng,
            train_state, train_state_sharding,
            data_sharding, replicated_sharding,
            mixed_loader, task_loaders, task_idx,
            checkpoint_manager, replay_loader, buffer_lst,
        )

        if config.cl == "er":
            # this saves the current buffer into a replay buffer for ER 
            # -> To inject my method, I'd need to actaully do this at the start of the task 
            # -> and also modify the create_torch_data_loader_replay_buffer situation 
            dataset_lst_final = (
                dataset_lst[:task_idx + 1] # basically creating everything up to this point 
                if not config.no_resampling
                else dataset_lst[:task_idx + 2] # creating including the new task 
            )
            replay_loader = create_torch_data_loader_replay_buffer(
                dataset_lst_final,
                loader.data_config(),
                model_config=config.model,
                action_horizon=config.model.action_horizon,
                batch_size=config.batch_size,
                sharding=data_sharding,
                shuffle=True,
                num_batches=None,
                num_workers=config.num_workers,
                seed=config.seed,
                skip_norm_stats=False,
            )

    if config.learn_one_task_idx > -1: # this is how you learn a single task 
        _run_task(config.learn_one_task_idx)
    else:
        for task_idx in range(prev_task_idx + 1, n_tasks): # this is the main CL loop 
            _run_task(task_idx)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())

# remaining questions: 
# - how is the config created? 
# - 