import collections
import dataclasses
import logging
import math
import os
import pathlib
import pickle
from typing import Any, Deque, Dict, List, Optional, Tuple

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import DummyVectorEnv
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import SubprocVectorEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import time
import torch

LIBERO_DUMMY_ACTION = np.array([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # Execution mode
    sequence: bool = False

    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO environment-specific parameters
    task_suite_name: str = "libero_object"  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    eval_upto_task: int = 0  # Checkpoint was trained up to this task id (inclusive)
    num_parallel_envs: int = 1
    mujoco_gl: str = "osmesa"
    render_gpu_device_id: int = -1

    # Initial states on disk (same layout as libero Benchmark.get_task_init_states: torch.load of .pruned_init)
    # If empty, loads: get_libero_path("init_states") / task.problem_folder / task.init_states_file
    init_states_path: str = ""
    init_states_folder: str = ""

    # Output paths
    video_out_path: str = "videos"
    result_out_path: str = "result_summary.pkl"
    base_dir: str = ""
    seed: int = 7
    video_fps: int = 10
    save_failed_only: bool = False

    # Debugging
    debug_env_creation: bool = False
    debug_rollout_shapes: bool = False
    debug_worker_lifecycle: bool = False
    debug_step_summary: bool = False
    debug_summary_interval: int = 25

    # Batched websocket inference (dummy server implements this; real OpenPI server is separate work)
    use_batched_inference: bool = False

    def load_result_summary(self) -> dict:
        with open(self.result_out_path, "rb") as f:
            return pickle.load(f)


@dataclasses.dataclass
class EpisodeSlot:
    slot_idx: int
    task_id: int
    task_description: str
    episode_idx: int
    initial_state: np.ndarray
    obs: Optional[Dict[str, Any]] = None
    action_plan: Deque[np.ndarray] = dataclasses.field(default_factory=collections.deque)
    replay_images: List[np.ndarray] = dataclasses.field(default_factory=list)
    t: int = 0
    done: bool = False
    success: bool = False
    error: Optional[str] = None


@dataclasses.dataclass
class EpisodeResult:
    task_id: int
    task_description: str
    episode_idx: int
    success: bool
    replay_images: List[np.ndarray]
    error: Optional[str]


def _get_max_steps(task_suite_name: str) -> int:
    steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if task_suite_name not in steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return steps[task_suite_name]


def _resolve_init_states_file(task, args: Args) -> pathlib.Path:
    """Path to the torch-saved init state tensor for this task (see libero benchmark)."""
    if args.init_states_path:
        p = pathlib.Path(args.init_states_path)
        assert p.is_file(), f"init_states_path is not a file: {p}"
        return p
    root = pathlib.Path(args.init_states_folder) if args.init_states_folder else pathlib.Path(get_libero_path("init_states"))
    p = root / task.problem_folder / task.init_states_file
    assert p.is_file(), f"Init states file not found: {p}"
    return p


def _load_initial_states_from_file(task, args: Args) -> np.ndarray:
    """Load all mujoco init states for a task; shape [N, ...]. Same source as Benchmark.get_task_init_states."""
    path = _resolve_init_states_file(task, args)
    raw = torch.load(str(path))
    arr = np.asarray(raw)
    assert arr.ndim >= 1, f"Expected init states with ndim >= 1, got shape {arr.shape} from {path}"
    n_envs_in_file = arr.shape[0]
    assert args.num_trials_per_task <= n_envs_in_file, (
        f"num_trials_per_task={args.num_trials_per_task} exceeds number of init states in {path} (N={n_envs_in_file})."
    )
    logging.info("Loaded %s init states from %s", n_envs_in_file, path)
    return arr


def _get_libero_env_args(task, resolution: int, render_gpu_device_id: int) -> Tuple[Dict[str, Any], str]:
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "render_gpu_device_id": render_gpu_device_id,
    }
    return env_args, task_description


def _make_env_factory(
    env_args: Dict[str, Any],
    seed: int,
    debug_env_creation: bool,
    slot_idx: int,
    mujoco_gl: str,
):
    def make_env():
        os.environ["MUJOCO_GL"] = mujoco_gl
        if debug_env_creation:
            logging.info(
                "Creating LIBERO env for slot=%s bddl=%s seed=%s resolution=%s MUJOCO_GL=%s",
                slot_idx,
                env_args["bddl_file_name"],
                seed,
                env_args["camera_heights"],
                mujoco_gl,
            )
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)
        return env

    return make_env


def _create_vector_env(task, args: Args):
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    env_args, task_description = _get_libero_env_args(task, LIBERO_ENV_RESOLUTION, args.render_gpu_device_id)
    env_fns = [
        _make_env_factory(env_args, args.seed + slot_idx, args.debug_env_creation, slot_idx, args.mujoco_gl)
        for slot_idx in range(args.num_parallel_envs)
    ]
    vector_env_cls = DummyVectorEnv if args.num_parallel_envs == 1 else SubprocVectorEnv
    env = vector_env_cls(env_fns)
    if args.debug_worker_lifecycle:
        logging.info(
            "Created %s with env_num=%s for task=%s",
            vector_env_cls.__name__,
            args.num_parallel_envs,
            task_description,
        )
    return env, task_description


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _preprocess_images(obs: Dict[str, Any], resize_size: int) -> Tuple[np.ndarray, np.ndarray]:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    return img, wrist_img


def _make_policy_input(obs: Dict[str, Any], img: np.ndarray, wrist_img: np.ndarray, task_description: str) -> Dict[str, Any]:
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }


def _infer_action_chunk(
    client: _websocket_client_policy.WebsocketClientPolicy,
    element: Dict[str, Any],
    replan_steps: int,
) -> np.ndarray:
    action_chunk = np.asarray(client.infer(element)["actions"], dtype=np.float32)
    assert len(action_chunk) >= replan_steps, (
        f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
    )
    assert action_chunk.ndim == 2, f"Expected action chunk shape [T, A], got {action_chunk.shape}"
    assert action_chunk.shape[1] == 7, f"Expected action dimension 7, got {action_chunk.shape}"
    return action_chunk[:replan_steps]


def _infer_action_chunk_batch(
    client: _websocket_client_policy.WebsocketClientPolicy,
    elements: List[Dict[str, Any]],
    replan_steps: int,
) -> np.ndarray:
    assert len(elements) >= 1
    batch = client.infer_batch(elements)
    actions = np.asarray(batch["actions"], dtype=np.float32)
    assert actions.ndim == 3, f"Expected batched actions [B, T, A], got {actions.shape}"
    assert actions.shape[0] == len(elements), f"B mismatch: got actions {actions.shape[0]} vs {len(elements)} elements"
    assert actions.shape[2] == 7, f"Expected action dimension 7, got {actions.shape}"
    assert actions.shape[1] >= replan_steps, (
        f"We want to replan every {replan_steps} steps, but policy only predicts {actions.shape[1]} steps."
    )
    return actions[:, :replan_steps, :]


def _episode_video_name(task_description: str, episode_idx: int, success: bool) -> str:
    suffix = "success" if success else "failure"
    task_segment = task_description.replace(" ", "_")
    return f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4"


def _write_rollout_video(video_dir: pathlib.Path, result: EpisodeResult, fps: int, save_failed_only: bool) -> None:
    if save_failed_only and result.success:
        return
    out_video = video_dir / _episode_video_name(result.task_description, result.episode_idx, result.success)
    imageio.mimwrite(out_video, [np.asarray(x) for x in result.replay_images], fps=fps)


def _log_obs_shapes_once(obs: Dict[str, Any], element: Dict[str, Any], task_id: int, slot_idx: int) -> None:
    logging.info(
        "Task %s slot %s obs shapes: agent=%s wrist=%s eef_pos=%s eef_quat=%s gripper=%s state=%s",
        task_id,
        slot_idx,
        obs["agentview_image"].shape,
        obs["robot0_eye_in_hand_image"].shape,
        obs["robot0_eef_pos"].shape,
        obs["robot0_eef_quat"].shape,
        obs["robot0_gripper_qpos"].shape,
        element["observation/state"].shape,
    )


def _validate_reset_obs(obs: Dict[str, Any], task_id: int, slot_idx: int) -> None:
    required_keys = {
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    }
    missing = required_keys.difference(obs.keys())
    assert not missing, f"Missing expected keys for task {task_id} slot {slot_idx}: {sorted(missing)}"


def _initialize_slots(
    env,
    task_id: int,
    task_description: str,
    initial_states: np.ndarray,
    episode_indices: List[int],
    args: Args,
    *,
    reset_workers: bool,
) -> List[EpisodeSlot]:
    active_ids = list(range(len(episode_indices)))
    print(episode_indices)
    n_states = initial_states.shape[0]
    for idx in episode_indices:
        assert 0 <= idx < n_states, f"episode index {idx} out of range for init states [0, {n_states})"
    init_batch = np.asarray([initial_states[idx] for idx in episode_indices])
    if args.debug_worker_lifecycle:
        logging.info(
            "Init slots task=%s episode_indices=%s reset_workers=%s",
            task_id,
            episode_indices,
            reset_workers,
        )
    if reset_workers:
        try:
            env.reset(id=active_ids)
        except Exception as exc:
            raise RuntimeError(
                "Failed to create/reset parallel LIBERO environments. "
                "This often means offscreen EGL context allocation failed for the requested "
                f"num_parallel_envs={len(active_ids)} on render_gpu_device_id={args.render_gpu_device_id}. "
                "Try a smaller num_parallel_envs, a different render_gpu_device_id, or debug with num_parallel_envs=1."
            ) from exc

    try:
        obs_batch = env.set_init_state(init_batch, id=active_ids)
        print("Resetted the enviornments based on the supplied state!")
    except Exception as exc:
        raise RuntimeError(
            "Failed while assigning initial states to parallel LIBERO environments. "
            "Check that the worker processes stayed alive and that the chosen parallelism fits the machine's "
            "offscreen rendering capacity."
        ) from exc

    slots: List[EpisodeSlot] = []
    for local_idx, episode_idx in enumerate(episode_indices):
        obs = obs_batch[local_idx]
        _validate_reset_obs(obs, task_id, local_idx)
        slots.append(
            EpisodeSlot(
                slot_idx=local_idx,
                task_id=task_id,
                task_description=task_description,
                episode_idx=episode_idx,
                initial_state=init_batch[local_idx],
                obs=obs,
            )
        )
    return slots


def _warmup_slots(env, slots: List[EpisodeSlot], args: Args) -> None:
    if args.num_steps_wait == 0:
        return

    active_ids = [slot.slot_idx for slot in slots if not slot.done]
    if not active_ids:
        return

    dummy_actions = np.repeat(LIBERO_DUMMY_ACTION[None, :], len(active_ids), axis=0)
    for warmup_step in range(args.num_steps_wait):
        if args.debug_step_summary:
            logging.info("Warmup step %s/%s for slots=%s", warmup_step + 1, args.num_steps_wait, active_ids)
        obs_batch, _, _, _ = env.step(dummy_actions, id=active_ids)
        for local_idx, slot_idx in enumerate(active_ids):
            slots[slot_idx].obs = obs_batch[local_idx]


def _run_parallel_episode_chunk(
    env,
    client: _websocket_client_policy.WebsocketClientPolicy,
    slots: List[EpisodeSlot],
    task_id: int,
    max_steps: int,
    args: Args,
) -> List[EpisodeResult]:
    _warmup_slots(env, slots, args)

    logged_shapes = False
    active_ids = [slot.slot_idx for slot in slots if not slot.done]
    while active_ids:
        infer_ids = [slot_idx for slot_idx in active_ids if not slots[slot_idx].action_plan]
        if infer_ids:
            elements: List[Dict[str, Any]] = []
            for slot_idx in infer_ids:
                slot = slots[slot_idx]
                assert slot.obs is not None
                img, wrist_img = _preprocess_images(slot.obs, args.resize_size)
                slot.replay_images.append(img)
                element = _make_policy_input(slot.obs, img, wrist_img, slot.task_description)
                if args.debug_rollout_shapes and not logged_shapes:
                    _log_obs_shapes_once(slot.obs, element, task_id, slot_idx)
                    logged_shapes = True
                elements.append(element)

            try:
                if args.use_batched_inference:
                    batch_actions = _infer_action_chunk_batch(client, elements, args.replan_steps)
                    for row, slot_idx in enumerate(infer_ids):
                        slots[slot_idx].action_plan.extend(batch_actions[row])
                else:
                    for element, slot_idx in zip(elements, infer_ids):
                        action_chunk = _infer_action_chunk(client, element, args.replan_steps)
                        slots[slot_idx].action_plan.extend(action_chunk)
            except Exception as exc:
                for slot_idx in infer_ids:
                    slot = slots[slot_idx]
                    slot.done = True
                    slot.success = False
                    slot.error = f"infer failed: {exc}"
                logging.exception(
                    "Inference failed for task=%s slots=%s",
                    task_id,
                    infer_ids,
                )

        active_ids = [slot.slot_idx for slot in slots if not slot.done]
        if not active_ids:
            break

        ready_ids = [slot_idx for slot_idx in active_ids if slots[slot_idx].action_plan]
        if not ready_ids:
            break

        action_batch = np.stack([np.asarray(slots[slot_idx].action_plan.popleft(), dtype=np.float32) for slot_idx in ready_ids])
        assert action_batch.shape[1] == 7, f"Expected per-step action shape [B, 7], got {action_batch.shape}"

        try:
            obs_batch, reward_batch, done_batch, _info = env.step(action_batch, id=ready_ids)
        except Exception as exc:
            raise RuntimeError(f"Vector env step failed for task={task_id} active_slots={ready_ids}") from exc

        finished_ids: List[int] = []
        for local_idx, slot_idx in enumerate(ready_ids):
            slot = slots[slot_idx]
            slot.obs = obs_batch[local_idx]
            slot.t += 1
            done = bool(done_batch[local_idx])
            reward = reward_batch[local_idx]
            if args.debug_step_summary and (
                slot.t == 1 or slot.t % args.debug_summary_interval == 0 or done or slot.t >= max_steps
            ):
                logging.info(
                    "Task=%s slot=%s episode=%s t=%s reward=%s done=%s remaining_plan=%s",
                    task_id,
                    slot.slot_idx,
                    slot.episode_idx,
                    slot.t,
                    reward,
                    done,
                    len(slot.action_plan),
                )

            if done:
                slot.done = True
                slot.success = True
                finished_ids.append(slot_idx)
                continue

            if slot.t >= max_steps:
                slot.done = True
                slot.success = False
                finished_ids.append(slot_idx)

        active_ids = [slot_idx for slot_idx in active_ids if slot_idx not in finished_ids and not slots[slot_idx].done]

    return [
        EpisodeResult(
            task_id=slot.task_id,
            task_description=slot.task_description,
            episode_idx=slot.episode_idx,
            success=slot.success,
            replay_images=slot.replay_images,
            error=slot.error,
        )
        for slot in slots
    ]


def _run_task_parallel(
    env,
    client: _websocket_client_policy.WebsocketClientPolicy,
    task_id: int,
    task_description: str,
    initial_states: np.ndarray,
    episode_indices: List[int],
    max_steps: int,
    args: Args,
) -> List[EpisodeResult]:
    results: List[EpisodeResult] = []
    if not episode_indices:
        return results

    chunk_starts = list(range(0, len(episode_indices), args.num_parallel_envs))
    for chunk_idx, start in enumerate(
        tqdm.tqdm(chunk_starts, leave=False, desc=f"Task {task_id} chunks")
    ):
        batch_episode_indices = episode_indices[start : start + args.num_parallel_envs]
        reset_workers = chunk_idx == 0
        slots = _initialize_slots(
            env,
            task_id,
            task_description,
            initial_states,
            batch_episode_indices,
            args,
            reset_workers=reset_workers,
        )
        results.extend(_run_parallel_episode_chunk(env, client, slots, task_id, max_steps, args))
    return results


def _resolve_output_paths(args: Args) -> Tuple[pathlib.Path, pathlib.Path]:
    base = pathlib.Path(args.base_dir) if args.base_dir else None
    video_dir = (base / args.video_out_path) if base else pathlib.Path(args.video_out_path)
    result_path = (base / args.result_out_path) if base else pathlib.Path(args.result_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    return video_dir, result_path


def _assert_batch_inference_supported(client: _websocket_client_policy.WebsocketClientPolicy) -> None:
    meta = client.get_server_metadata()
    assert meta.get("supports_batch_infer") is True, (
        "Batched inference was requested, but the connected server metadata does not advertise "
        f"'supports_batch_infer'. Metadata: {meta!r}. "
        "Use examples/libero/dummy_policy_server.py for now, or implement batching in the real server."
    )


def eval_libero_parallel(args: Args) -> None:
    np.random.seed(args.seed)
    assert args.num_parallel_envs >= 1, "num_parallel_envs must be >= 1"

    video_dir, _ = _resolve_output_paths(args)
    max_steps = _get_max_steps(args.task_suite_name)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    if args.use_batched_inference:
        _assert_batch_inference_supported(client)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        beg = time.time()
        task = task_suite.get_task(task_id)
        initial_states = _load_initial_states_from_file(task, args)
        env, task_description = _create_vector_env(task, args)
        try:
            episode_indices = list(range(args.num_trials_per_task))
            task_results = _run_task_parallel(
                env,
                client,
                task_id,
                task_description,
                initial_states,
                episode_indices,
                max_steps,
                args,
            )
        finally:
            env.close()
        print(f"Time taken for this environment is {time.time() - beg}")

        task_successes = 0
        for result in task_results:
            task_successes += int(result.success)
            total_successes += int(result.success)
            total_episodes += 1
            _write_rollout_video(video_dir, result, args.video_fps, args.save_failed_only)
            if result.error:
                logging.error(
                    "Task=%s episode=%s failed with error: %s",
                    result.task_id,
                    result.episode_idx,
                    result.error,
                )

        logging.info(
            "Task %s success rate: %.4f (%s/%s)",
            task_id,
            task_successes / float(len(task_results)),
            task_successes,
            len(task_results),
        )
        logging.info(
            "Cumulative success rate: %.4f (%s/%s)",
            total_successes / float(total_episodes),
            total_successes,
            total_episodes,
        )

    logging.info("Total success rate: %.4f", total_successes / float(total_episodes))
    logging.info("Total episodes: %s", total_episodes)


def eval_libero_sequence_parallel(args: Args) -> None:
    np.random.seed(args.seed)
    assert args.num_parallel_envs >= 1, "num_parallel_envs must be >= 1"

    video_dir, result_path = _resolve_output_paths(args)
    max_steps = _get_max_steps(args.task_suite_name)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    if args.use_batched_inference:
        _assert_batch_inference_supported(client)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_total = task_suite.n_tasks
    assert 0 <= args.eval_upto_task < n_total, f"eval_upto_task must be < {n_total}, got {args.eval_upto_task}"

    if result_path.is_file():
        with open(result_path, "rb") as f:
            result_summary = pickle.load(f)
        logging.info("Loaded existing result_summary from %s", result_path)
    else:
        result_summary = {
            "S_conf_mat": np.zeros((n_total, n_total)),
            "L_conf_mat": np.zeros((n_total, n_total)),
            "S_fwd": np.zeros(n_total),
            "L_fwd": np.zeros(n_total),
        }
        logging.info("Creating fresh result_summary at %s", result_path)

    row = args.eval_upto_task
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(row + 1), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = _load_initial_states_from_file(task, args)
        env, task_description = _create_vector_env(task, args)
        try:
            episode_indices = list(range(args.num_trials_per_task))
            task_results = _run_task_parallel(
                env,
                client,
                task_id,
                task_description,
                initial_states,
                episode_indices,
                max_steps,
                args,
            )
        finally:
            env.close()

        task_successes = 0
        task_losses = []
        for result in task_results:
            task_successes += int(result.success)
            total_successes += int(result.success)
            total_episodes += 1
            task_losses.append(0.0 if result.success else 1.0)
            _write_rollout_video(video_dir, result, args.video_fps, args.save_failed_only)
            if result.error:
                logging.error(
                    "Task=%s episode=%s failed with error: %s",
                    result.task_id,
                    result.episode_idx,
                    result.error,
                )

        success_rate = task_successes / float(args.num_trials_per_task)
        avg_loss = float(np.mean(task_losses))
        logging.info("[Task %s] success-rate=%.3f, loss=%.3f", task_id, success_rate, avg_loss)
        result_summary["S_conf_mat"][row, task_id] = success_rate
        result_summary["L_conf_mat"][row, task_id] = avg_loss

    result_summary["S_fwd"][row] = result_summary["S_conf_mat"][row, row]
    result_summary["L_fwd"][row] = result_summary["L_conf_mat"][row, row]

    with open(result_path, "wb") as f:
        pickle.dump(result_summary, f)
    logging.info("Saved result_summary to %s", result_path)
    logging.info("Forward success row: %s", result_summary["S_conf_mat"][row, : row + 1])
    logging.info("Forward loss row   : %s", result_summary["L_conf_mat"][row, : row + 1])
    logging.info(
        "Cumulative success rate: %.4f (%s/%s)",
        total_successes / float(total_episodes),
        total_successes,
        total_episodes,
    )


def main(args: Args) -> None:
    if args.sequence:
        eval_libero_sequence_parallel(args)
    else:
        eval_libero_parallel(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
