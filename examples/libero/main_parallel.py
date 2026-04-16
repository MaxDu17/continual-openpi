import matplotlib
matplotlib.use("Agg")  # prevent debugpy from trying to activate a GUI backend (no tkinter in this venv)

import collections
import dataclasses
import datetime
import glob
import logging
import math
import multiprocessing
import os
import pathlib

import imageio
import yaml
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.benchmark import Benchmark, register_benchmark, grab_language_from_filename, Task
from libero.libero.benchmark import BENCHMARK_MAPPING as LIBERO_BENCHMARK_MAPPING
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,   # longest training demo has 254 steps
    "libero_goal": 300,     # longest training demo has 270 steps
    "libero_10": 520,       # longest training demo has 505 steps
    "libero_90": 400,       # longest training demo has 373 steps
}


def _get_hdf5_files(base_dir, splits):
    files = []
    for split in splits:
        files.extend(glob.glob(os.path.join(base_dir, split, "*.hdf5")))
    return sorted(files)


class _NewBenchmark(Benchmark):
    def __init__(self, split):
        super().__init__(task_order_index=0)
        self.name = split
        self.hdf5_files = _get_hdf5_files(get_libero_path("datasets"), [split])
        self._make_benchmark()

    def _make_benchmark(self):
        task_names = [os.path.basename(x).replace("_demo.hdf5", "") for x in self.hdf5_files]
        tasks = {}
        for task in task_names:
            language = grab_language_from_filename(task + ".bddl")
            tasks[task] = Task(
                name=task,
                language=language,
                problem="Libero",
                problem_folder=self.name,
                bddl_file=f"{task}.bddl",
                init_states_file=f"{task}.pruned_init",
            )
        self.tasks = sorted(tasks.values(), key=lambda x: x.name)
        self.n_tasks = len(self.tasks)


def _create_benchmark_class(split_name):
    class BenchmarkClass(_NewBenchmark):
        def __init__(self):
            super().__init__(split_name)
    BenchmarkClass.__name__ = split_name.upper()
    BenchmarkClass.__qualname__ = split_name.upper()
    return register_benchmark(BenchmarkClass)


def _discover_and_register_benchmarks():
    """Auto-discover split directories in the libero datasets folder and register them."""
    libero_datasets_dir = get_libero_path("datasets")
    if not os.path.exists(libero_datasets_dir):
        return
    existing = {name.lower() for name in LIBERO_BENCHMARK_MAPPING.keys()}
    new_splits = sorted(
        d for d in os.listdir(libero_datasets_dir)
        if os.path.isdir(os.path.join(libero_datasets_dir, d)) and d.lower() not in existing
    )
    for split_name in new_splits:
        _create_benchmark_class(split_name)
        logging.debug(f"Registered benchmark: {split_name}")


_discover_and_register_benchmarks()


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_object"  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    eval_upto_task: int = 0  # Evaluate tasks 0..eval_upto_task inclusive (checkpoint trained through this task id)
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    out_path: str = "data/libero/runs"  # Run logs / stats; each run uses a timestamped subfolder under this path
    base_dir: str = ""  # If set, out_path is resolved under this directory (same semantics as main.py)
    run_id: str = ""  # Optional label appended to the run folder name (e.g. "pi05_baseline")
    num_workers: int = 20  # Number of parallel worker processes

    seed: int = 7  # Random Seed (for reproducibility)


def _configure_worker_logging() -> None:
    """Spawned workers never run ``if __name__ == "__main__"``; enable INFO logs there too.

    ``WebsocketClientPolicy`` logs wait/retry messages at INFO; without this, workers look hung
    while they sleep in ``_wait_for_server`` (same loop as in the main process, but silent).
    """
    logging.basicConfig(level=logging.INFO)


def run_task(task_args: tuple) -> dict:
    """Unpack args and run a single task. Top-level for multiprocessing pickling."""
    task_id, args, max_steps, run_dir = task_args
    try:
        return _run_task_inner(task_id, args, max_steps, run_dir)
    except Exception as e:
        # Re-raise as plain RuntimeError so it can be pickled back to the main process
        raise RuntimeError(f"[Task {task_id}] failed: {e}") from None


def _run_task_inner(task_id: int, args: Args, max_steps: int, run_dir: pathlib.Path) -> dict:
    np.random.seed(args.seed + task_id)

    # Each worker initializes its own task suite, env, and client
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed + task_id)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    video_out_path = run_dir / "videos"
    task_episodes, task_successes = 0, 0
    saved_success, saved_failure = False, False

    for episode_idx in range(args.num_trials_per_task):
        logging.info(f"[Task {task_id}] Episode {episode_idx + 1}/{args.num_trials_per_task}: {task_description}")

        env.reset()
        action_plan = collections.deque()
        obs = env.set_init_state(initial_states[episode_idx])

        t = 0
        need_video = not saved_success or not saved_failure
        replay_images = [] if need_video else None
        done = False

        while t < max_steps + args.num_steps_wait:
            try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Get preprocessed image
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                if replay_images is not None:
                    replay_images.append(img)

                if not action_plan:
                    element = {
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

                    action_chunk = client.infer(element)["actions"]
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    break
                t += 1

            except Exception as e:
                logging.error(f"[Task {task_id}] Caught exception: {e}")
                break

        task_episodes += 1

        # Save at most one success and one failure video per task
        if replay_images is not None:
            task_segment = task_description.replace(" ", "_")
            if done and not saved_success:
                imageio.mimwrite(
                    video_out_path / f"task{task_id:02d}_{task_segment}_success.mp4",
                    [np.asarray(x[:, ::-1]) for x in replay_images],
                    fps=20,
                )
                saved_success = True
            elif not done and not saved_failure:
                imageio.mimwrite(
                    video_out_path / f"task{task_id:02d}_{task_segment}_failure.mp4",
                    [np.asarray(x[:, ::-1]) for x in replay_images],
                    fps=20,
                )
                saved_failure = True

        logging.info(f"[Task {task_id}] Episode {episode_idx + 1}: {'success' if done else 'failure'}")

    task_success_rate = float(task_successes) / float(task_episodes)
    logging.info(f"[Task {task_id}] Success rate: {task_success_rate:.2%}")
    return {
        "task_id": task_id,
        "task": task_description,
        "episodes": task_episodes,
        "successes": task_successes,
        "success_rate": round(task_success_rate, 4),
    }


def eval_suite(args: Args, task_suite_name: str, run_dir: pathlib.Path) -> dict:
    """Evaluate a single task suite and return its stats."""
    if task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    max_steps = MAX_STEPS[task_suite_name]

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {task_suite_name}")

    (run_dir / "videos").mkdir(parents=True, exist_ok=True)

    assert 0 <= args.eval_upto_task < num_tasks_in_suite, (
        f"eval_upto_task must be in [0, {num_tasks_in_suite}), got {args.eval_upto_task}"
    )
    num_tasks = args.eval_upto_task + 1
    task_args = [(task_id, args, max_steps, run_dir) for task_id in range(num_tasks)]

    num_workers = min(args.num_workers, num_tasks)
    logging.info(f"Running {num_tasks} tasks with {num_workers} workers")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=_configure_worker_logging) as pool:
        per_task_stats = list(
            tqdm.tqdm(pool.imap_unordered(run_task, task_args), total=num_tasks, desc=task_suite_name)
        )

    per_task_stats.sort(key=lambda s: s["task_id"])

    total_episodes = sum(s["episodes"] for s in per_task_stats)
    total_successes = sum(s["successes"] for s in per_task_stats)
    total_success_rate = float(total_successes) / float(total_episodes)

    logging.info(f"[{task_suite_name}] Success rate: {total_success_rate:.2%} ({total_successes}/{total_episodes})")

    stats = {
        "split": task_suite_name,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": round(total_success_rate, 4),
        "tasks": per_task_stats,
    }
    stats_path = run_dir / f"{task_suite_name}.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    logging.info(f"Stats saved to {stats_path}")
    return stats


def eval_libero(args: Args) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{args.run_id}" if args.run_id else timestamp
    base = pathlib.Path(args.base_dir) if args.base_dir else None
    runs_root = (base / args.out_path) if base else pathlib.Path(args.out_path)
    run_dir = runs_root / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\nRun directory: {run_dir }\n{'='*60}\n")

    suite_run_dir = run_dir / args.task_suite_name
    eval_suite(args, args.task_suite_name, suite_run_dir)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
