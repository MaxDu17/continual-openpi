from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils
import time

def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
            "buffer_lst": ocp.PyTreeCheckpointHandler(),
            "task_idx": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=None,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    task_loaders=None,
    task_idx=None,
    replay_loader=None,
    buffer_lst=None,
):
    
    if task_loaders is None:
        
        def save_assets(directory: epath.Path):
            # Save the normalization stats.
            data_config = data_loader.data_config()
            norm_stats = data_config.norm_stats
            if norm_stats is not None and data_config.asset_id is not None:
                _normalize.save(directory / data_config.asset_id, norm_stats)

        # Split params that can be used for inference into a separate item.
        with at.disable_typechecking():
            train_state, params = _split_params(state)
        items = {
            "assets": save_assets,
            "train_state": train_state,
            "params": {"params": params},
        }
        checkpoint_manager.save(step, items)

    else:
        
        def save_assets(directory: epath.Path):
            # Save the normalization stats.
            # Note: save the norm stats for all tasks individually.
            for data_loader in task_loaders:
                data_config = data_loader.data_config()
                norm_stats = data_config.norm_stats
                if norm_stats is not None and data_config.asset_id is not None:
                    _normalize.save(directory / data_config.asset_id, norm_stats)

        # Split params that can be used for inference into a separate item.
        with at.disable_typechecking():
            train_state, params = _split_params(state)
        items = {
            "assets": save_assets,
            "train_state": train_state,
            "params": {"params": params},
        }

        if buffer_lst is not None:
            items["buffer_lst"] = {"buffer_lst": buffer_lst}
        else:
            items["buffer_lst"] = {"buffer_lst": []}
        
        if task_idx is not None:
            items["task_idx"] = {"task_idx": task_idx}
        else:
            items["task_idx"] = {"task_idx": -1}
            
        success = checkpoint_manager.save(step, items)

        if task_idx is not None: 
            checkpoint_manager.wait_until_finished()        
            ckpt_path = epath.Path(checkpoint_manager.directory) / str(step)
            (ckpt_path / f"task_idx_{str(task_idx)}.txt").write_text(str(task_idx))


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
    restore_task_idx = False,
    restore_replay_buffer = False,
) -> training_utils.TrainState:
    del data_loader

    # restore train_state and params
    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        
        items={
            "train_state": train_state,
            "params": {"params": params},
        }
            
        try:
            restored = checkpoint_manager.restore(
                step, 
                items=items, 
            )
        
            train_state = _merge_params(
                restored["train_state"], restored["params"]
            )
        except ValueError as e:
            # Check for specific mismatch error related to opt_state
            # The error message typically contains "structure" and "match" or "Diff"
            if "opt_state" in str(e) and ("structure" in str(e) or "Diff" in str(e)):
                logging.warning(
                    "Optimizer state structure mismatch detected (likely due to changing Learning Rates / Optimizer). "
                    "Resetting optimizer state and restoring parameters/step only."
                )
                # Retry restoring params with target, but train_state without target (to get step)
                items_retry = {
                        "train_state": None, # Restore raw to extract step
                        "params": {"params": params}, # Restore params into target
                }
                restored = checkpoint_manager.restore(step, items=items_retry)
                
                # Extract step from raw train_state
                train_state_raw = restored["train_state"]
                # Assuming standard Flax serialization where dataclass becomes dict-like
                restored_step = train_state_raw.get("step") if isinstance(train_state_raw, dict) else getattr(train_state_raw, "step", None)
                
                if restored_step is None:
                        # Fallback if structure is unexpected
                        raise e
                
                # Update the original state (which has fresh opt_state) with restored step and params
                # We do NOT use the restored opt_state
                state_with_params = _merge_params(state, restored["params"])
                train_state = dataclasses.replace(state_with_params, step=restored_step)
            else:
                raise e
    
    extra_items = {}
    if restore_replay_buffer:
        extra_items["buffer_lst"] = None  # target not required if metadata exists
    if restore_task_idx:
        extra_items["task_idx"] = None
    
    buffer_lst = None
    task_idx = None
    if extra_items:
        # restore replay_state, task_idx
        restored_new = checkpoint_manager.restore(step, items=extra_items)
        if restore_replay_buffer:
            buffer_lst = restored_new.get("buffer_lst")["buffer_lst"]
        if restore_task_idx:
            task_idx = restored_new.get("task_idx")["task_idx"]
            
    return train_state, task_idx, buffer_lst


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        return None
        # raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
