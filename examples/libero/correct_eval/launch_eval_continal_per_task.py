import glob, os, re, sys, pathlib, stat
from pathlib import Path

BASE_DIR = sys.argv[1]

parent_name = Path(BASE_DIR).resolve().parent.name
TASK_SUITE = next(
    chunk for chunk in parent_name.split('-') if chunk.startswith('libero_')
)

if TASK_SUITE.endswith('_sequential'):
    TASK_SUITE = TASK_SUITE[:-len('_sequential')]

PORT = 8000
TRIALS = 50
CMD_DIR = pathlib.Path(BASE_DIR)
CMD_DIR.mkdir(exist_ok=True)

POLICY_CONFIG = pathlib.Path(BASE_DIR).parent.name


def discover_checkpoints():
    step_dirs = (d for d in glob.glob(os.path.join(BASE_DIR, "*")) if os.path.isdir(d))
    ckpts = []
    for d in step_dirs:
        step_name = os.path.basename(d)
        if step_name.isdigit() and step_name.endswith("999"):
            idx_file = glob.glob(os.path.join(d, "task_idx_*.txt"))
            if idx_file:
                task_idx = int(re.search(r"task_idx_(\d+)\.txt", idx_file[0]).group(1))
                ckpts.append((d, task_idx, int(step_name)))
    ckpts.sort(key=lambda x: x[2])
    return ckpts


def write_shell(name: pathlib.Path, lines: list[str]):
    name.write_text("\n".join(lines) + "\n")
    name.chmod(name.stat().st_mode | stat.S_IXUSR)


def main():
    for n, (ckpt_dir, task_idx, step) in enumerate(discover_checkpoints()):
        script = CMD_DIR / f"step_{step}.sh"

        lines = [
            "#!/usr/bin/env bash",
            f"python examples/libero/correct_eval/eval_continual_per_task.py "
            f"{BASE_DIR} {TASK_SUITE} {task_idx} {step}"
        ]
        write_shell(script, lines)
        print(f"Wrote {script}")


if __name__ == "__main__":
    main()
