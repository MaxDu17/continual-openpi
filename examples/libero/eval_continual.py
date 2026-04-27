#!/usr/bin/env python3
"""
eval_continual.py

Iterates through checkpoint folders, starts the policy server,
evaluates up to the task index recorded in `task_idx_*.txt`,
then shuts the server down before moving on to the next checkpoint.
"""

import glob
import http.client
import os
import re
import signal
import socket
import subprocess
import sys
import time

# Configuration
BASE_DIR = sys.argv[1]
POLICY_CONFIG = BASE_DIR.split("/")[-2]
# TODO: THIS IS NOT ROBUST 
assert len(BASE_DIR.split("/")[-1]) > 0, "Don't put a '/' at the end of your folder!" 

TASK_SUITE = sys.argv[2]

def find_unused_port():
    """Find an unused port by binding a socket to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

PORT = find_unused_port()
WAIT_FOR_BOOT = 240
TRIALS_PER_TASK = 50
CKPT_INTERVAL = 1000
SHUTDOWN_TIMEOUT = 240
PORT_WAIT_TIMEOUT = 240
POLL_INTERVAL = 1 #0.25


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
        "--task_idx", str(task_idx), # not 100% sure what the task idx does 
        "policy:checkpoint",
        "--policy.config", POLICY_CONFIG,
        "--policy.dir", ckpt_dir,
    ]

    # cmd = [
    #     "uv", "run", "scripts/serve_policy.py",
    #     "--port", str(PORT),
    #     "--env", "LIBERO",
    #     "policy:default"
    # ]
    return subprocess.Popen(cmd, env=env, start_new_session=True) #,
                            # stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def healthz_ok(port):
    """True if policy server responds OK on GET /healthz (same path as websocket_policy_server)."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
        conn.request("GET", "/healthz")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status == 200
    except OSError:
        return False


def wait_for_port_state(port, should_be_healthy, timeout_s, state_name):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if healthz_ok(port) == should_be_healthy:
            return
        time.sleep(POLL_INTERVAL)
    assert False, f"Timed out waiting for port {port} to become {state_name}"


def wait_for_server_boot(srv):
    print(f"[driver] Waiting up to {WAIT_FOR_BOOT}s for server to bind on port {PORT}")
    deadline = time.time() + WAIT_FOR_BOOT
    while time.time() < deadline:
        exit_code = srv.poll()
        print("Polling server...")
        assert exit_code is None, f"Policy server exited during boot with exit code {exit_code}"
        if healthz_ok(PORT):
            print(f"[driver] Server is up on port {PORT} (healthz OK)")
            return
        time.sleep(POLL_INTERVAL)
    assert False, f"Policy server did not bind to port {PORT} within {WAIT_FOR_BOOT}s"


def stop_server(srv):
    if srv.poll() is not None:
        print(f"[driver] Server already exited with code {srv.returncode}")
    else:
        pgid = os.getpgid(srv.pid)
        print(f"[driver] Sending SIGINT to policy server process group {pgid}")
        os.killpg(pgid, signal.SIGINT)
        try:
            srv.wait(timeout=SHUTDOWN_TIMEOUT)
            print(f"[driver] Server exited with code {srv.returncode}")
        except subprocess.TimeoutExpired:
            print(f"[driver] Process group {pgid} did not exit after SIGINT; sending SIGKILL")
            os.killpg(pgid, signal.SIGKILL)
            srv.wait(timeout=5)
            print(f"[driver] Server killed; exit code {srv.returncode}")
    print(f"[driver] Waiting for port {PORT} to be released")
    wait_for_port_state(
        PORT,
        should_be_healthy=False,
        timeout_s=PORT_WAIT_TIMEOUT,
        state_name="unreachable (no healthz)",
    )


def run_eval(task_idx, ckpt_dir):
    print(f"[driver] Evaluating (eval_upto_task={task_idx}) with checkpoint: {ckpt_dir}")

    env = os.environ.copy()
    cmd = [
        "python", "examples/libero/main_parallel.py",
        "--args.task_suite_name", TASK_SUITE,
        "--args.eval_upto_task", str(task_idx),
        "--args.num_trials_per_task", str(TRIALS_PER_TASK),
        "--args.port", str(PORT),
        "--args.base_dir", ckpt_dir,
        "--args.num_trials_per_task", str(50)
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
    import time

    ckpts = discover_checkpoints()
    if not ckpts:
        print("No checkpoints found under", BASE_DIR)
        return
    print("Found", len(ckpts), "checkpoints")

    overall_start_time = time.time()
    per_ckpt_times = []

    for ckpt_dir, task_idx, step in ckpts:
        print(f"\n=== step {step:6d}  (task_idx={task_idx}) ===")
        assert not healthz_ok(PORT), f"Port {PORT} already serves /healthz before launch"
        ckpt_start_time = time.time()
        srv = launch_server(ckpt_dir, task_idx)
        wait_for_server_boot(srv)
        try:
            run_eval(task_idx, ckpt_dir)
        finally:
            stop_server(srv)
        ckpt_end_time = time.time()
        elapsed = ckpt_end_time - ckpt_start_time
        per_ckpt_times.append((step, task_idx, elapsed))
        print(f"Completed step {step} in {elapsed:.2f} seconds")

    overall_end_time = time.time()
    overall_elapsed = overall_end_time - overall_start_time

    print("\n===== Timing summary =====")
    for step, task_idx, elapsed in per_ckpt_times:
        print(f"Step {step:6d} (task_idx={task_idx}): {elapsed:.2f} seconds")
    print(f"\nTotal time for all checkpoints: {overall_elapsed:.2f} seconds")
   

if __name__ == "__main__":
    main()
