#!/usr/bin/env python3
"""
eval_continual.py

Iterates through checkpoint folders, starts the policy server,
evaluates up to the task index recorded in `task_idx_*.txt`,
then shuts the server down before moving on to the next checkpoint.
"""

import glob, os, re, subprocess, time, signal, sys, pathlib

# Configuration
BASE_DIR = sys.argv[1]
POLICY_CONFIG = BASE_DIR.split("/")[-2]
# TODO: THIS IS NOT ROBUST 
assert len(BASE_DIR.split("/")[-1]) > 0, "Don't put a '/' at the end of your folder!" 

TASK_SUITE = sys.argv[2]

PORT = 8000
WAIT_FOR_BOOT = 10
TRIALS_PER_TASK = 50
CKPT_INTERVAL = 1000


def discover_checkpoints():
    """Return sorted list of (ckpt_dir, task_idx, step)."""
    step_dirs = [d for d in glob.glob(os.path.join(BASE_DIR, "*")) if os.path.isdir(d)]
    ckpts = []
    for d in step_dirs:
        step_name = os.path.basename(d)
        if step_name.isdigit() and step_name.endswith("9999"):
            idx_files = glob.glob(os.path.join(d, "task_idx_*.txt"))
            if not idx_files:
                print(f"[skip] {d} – no task_idx_*.txt") # , file=sys.stderr)
                continue
            task_idx = int(re.search(r"task_idx_(\d+)\.txt", idx_files[0]).group(1))
            ckpts.append((d, task_idx, int(step_name)))
    ckpts.sort(key=lambda x: x[2])
    return ckpts


def launch_server(ckpt_dir, task_idx):
    print(f"[driver] Launching policy server from checkpoint: {ckpt_dir}")

    env = os.environ.copy()
    cmd = [
        "uv", "run", "scripts/serve_policy.py",
        "--port", str(PORT),
        "--task_idx", str(task_idx),
        "policy:checkpoint",
        "--policy.config", POLICY_CONFIG,
        "--policy.dir", ckpt_dir,
    ]
    return subprocess.Popen(cmd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def run_eval(task_idx, ckpt_dir):
    print(f"[driver] Evaluating (eval_upto_task={task_idx}) with checkpoint: {ckpt_dir}")

    env = os.environ.copy()
    cmd = [
        "python", "examples/libero/main_parallel.py",
        "--args.task_suite_name", TASK_SUITE,
        "--args.eval_upto_task", str(task_idx),
        "--args.num_trials_per_task", str(TRIALS_PER_TASK),
        "--args.base_dir", ckpt_dir,
    ]
    subprocess.run(cmd, env=env, check=True)

    # print(env)
    # cmd = [
    #     "python", "-c", "import sys; print(f'Python version: {sys.version}'); print(f'Executable: {sys.executable}')"
    # ]
    # cmd = [
    #     "python", "-c", "import sys; print(f'Python version: {sys.version}'); print(f'Executable: {sys.executable}')"
    # ]
    # subprocess.run(cmd, env=env, check=True)
    # import robosuite.utils.transform_utils as T


def main():
    ckpts = discover_checkpoints()
    if not ckpts:
        print("No checkpoints found under", BASE_DIR)
        return
    print("Found", len(ckpts), "checkpoints")

    for ckpt_dir, task_idx, step in ckpts:
        print(f"\n=== step {step:6d}  (task_idx={task_idx}) ===")
        srv = launch_server(ckpt_dir, task_idx)
        time.sleep(WAIT_FOR_BOOT)
        try:
            run_eval(task_idx, ckpt_dir)
        finally:
            srv.send_signal(signal.SIGINT)
            try:
                srv.wait(timeout=30)
            except subprocess.TimeoutExpired:
                srv.kill()
        print(f"Completed step {step}")


if __name__ == "__main__":
    main()
