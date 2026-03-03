#!/usr/bin/env python3
"""
eval_continual_per_task_single.py

Evaluates a single task at a given checkpoint step.
Starts the policy server, runs evaluation, then shuts down.
"""

import glob, os, re, subprocess, time, signal, sys, pathlib

BASE_DIR = sys.argv[1]
POLICY_CONFIG = BASE_DIR.split("/")[-2]
TASK_SUITE = sys.argv[2]

PORT = 8000
WAIT_FOR_BOOT = 10
TRIALS_PER_TASK = 50

EVAL_SINGLE_TASK = True


def launch_server(ckpt_dir, task_idx, port):
    print(f"[driver] Launching policy server from checkpoint: {ckpt_dir}")

    env = os.environ.copy()

    if task_idx == -1:
        cmd = [
            "uv", "run", "scripts/serve_policy.py",
            "--port", str(port),
            "policy:checkpoint",
            "--policy.config", POLICY_CONFIG,
            "--policy.dir", ckpt_dir,
        ]
    else:
        policy_config = POLICY_CONFIG
        if "-b16" in ckpt_dir:
            policy_config = POLICY_CONFIG + "-B/16"

        cmd = [
            "uv", "run", "scripts/serve_policy.py",
            "--port", str(port),
            "--task_idx", str(task_idx),
            "policy:checkpoint",
            "--policy.config", policy_config,
            "--policy.dir", ckpt_dir,
        ]

    print(f"Server Command: {' '.join(cmd)}")

    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def run_eval_multitask(ckpt_dir, port):
    print(f"[driver] Evaluating with checkpoint: {ckpt_dir}")

    env = os.environ.copy()
    cmd = [
        "python", "examples/libero/main.py",
        "--args.task_suite_name", TASK_SUITE,
        "--args.num_trials_per_task", str(TRIALS_PER_TASK),
        "--args.base_dir", ckpt_dir,
        "--args.port", str(port),
    ]

    print(f"Inference Command: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def run_eval(task_idx, ckpt_dir, port):
    print(f"[driver] Evaluating (eval_upto_task={task_idx}) with checkpoint: {ckpt_dir}")

    env = os.environ.copy()
    cmd = [
        sys.executable, "examples/libero/main_single.py",
        "--args.task_suite_name", TASK_SUITE,
        "--args.eval_task_idx", str(task_idx),
        "--args.num_trials_per_task", str(TRIALS_PER_TASK),
        "--args.base_dir", ckpt_dir,
        "--args.port", str(port),
    ]

    print(f"Inference Command: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def shutdown_server(proc, timeout=10):
    """Try SIGINT -> SIGTERM -> SIGKILL on the whole process group."""
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGINT)

    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    os.killpg(pgid, signal.SIGTERM)

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait()


def main():
    base_dir, task_idx, step = BASE_DIR, int(sys.argv[3]), int(sys.argv[4])
    ckpt_dir = os.path.join(BASE_DIR, sys.argv[4])
    port = PORT

    if task_idx == -1:
        srv = launch_server(ckpt_dir, task_idx, port)
        time.sleep(WAIT_FOR_BOOT)
        try:
            run_eval_multitask(ckpt_dir, port)
        finally:
            shutdown_server(srv)
    else:
        start_idx = task_idx if EVAL_SINGLE_TASK else 0

        for task_idx_j in range(start_idx, task_idx + 1):
            print(f"\n=== step {step:6d}  (task_idx={task_idx}) ===")
            srv = launch_server(ckpt_dir, task_idx_j, port)
            time.sleep(WAIT_FOR_BOOT)
            try:
                run_eval(task_idx_j, ckpt_dir, port)
            finally:
                shutdown_server(srv)
            print(f"Completed step {step} for task {task_idx_j}")
            port += 1


if __name__ == "__main__":
    main()
