import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import pickle
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
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

    # Output paths
    video_out_path: str = "videos"
    result_out_path: str = "result_summary.pkl"
    base_dir: str = ""
    seed: int = 7

    def load_result_summary(self) -> dict:
        """Load the result summary from the specified path."""
        with open(self.result_out_path, "rb") as f:
            return pickle.load(f)

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


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    base = pathlib.Path(args.base_dir) if args.base_dir else None
    video_dir = (base / args.video_out_path) if base else pathlib.Path(args.video_out_path)
    result_path = (base / args.result_out_path) if base else pathlib.Path(args.result_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    max_steps = _get_max_steps(args.task_suite_name)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

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
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(video_dir) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def eval_libero_sequence(args: Args) -> None:
    """Evaluate tasks 0..eval_upto_task inclusive and compute lifelong-learning metrics."""
    base = pathlib.Path(args.base_dir) if args.base_dir else None
    video_dir = (base / args.video_out_path) if base else pathlib.Path(args.video_out_path)
    result_path = (base / args.result_out_path) if base else pathlib.Path(args.result_out_path)

    np.random.seed(args.seed)
    video_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_total = task_suite.n_tasks
    assert 0 <= args.eval_upto_task < n_total, (
        f"eval_upto_task must be < {n_total}, got {args.eval_upto_task}")

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
    max_steps = _get_max_steps(args.task_suite_name)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(row + 1), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes = 0
        task_successes, task_losses = 0, []
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), leave=False, desc=f"Task {task_id}"):

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t, done = 0, False
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, *_ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"Policy predicted {len(action_chunk)} steps < replan_steps={args.replan_steps}")
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, _info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Caught exception in task {task_id}, episode {episode_idx}: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_seg = task_description.replace(" ", "_")
            out_video = pathlib.Path(video_dir) / f"rollout_{task_seg}_{episode_idx}_{suffix}.mp4"
            imageio.mimwrite(out_video, [np.asarray(x) for x in replay_images], fps=10)

            task_losses.append(0.0 if done else 1.0)

        success_rate = task_successes / float(args.num_trials_per_task)
        avg_loss = float(np.mean(task_losses))
        logging.info(f"[Task {task_id}] success-rate={success_rate:.3f}, loss={avg_loss:.3f}")

        row = args.eval_upto_task
        result_summary["S_conf_mat"][row, task_id] = success_rate
        result_summary["L_conf_mat"][row, task_id] = avg_loss

    row = args.eval_upto_task
    result_summary["S_fwd"][row] = result_summary["S_conf_mat"][row, row]
    result_summary["L_fwd"][row] = result_summary["L_conf_mat"][row, row]

    with open(result_path, "wb") as f:
        pickle.dump(result_summary, f)
    logging.info(f"Saved result_summary to {result_path}")

    logging.info("Forward success row: %s", result_summary["S_conf_mat"][row, : row + 1])
    logging.info("Forward loss row   : %s", result_summary["L_conf_mat"][row, : row + 1])


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
