"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset is written under `--output-dir` if set, otherwise under the LeRobot home
directory (see `LEROBOT_HOME` / `HF_LEROBOT_HOME`).
Running this conversion script will take approximately 30 minutes.
"""

import pathlib
import re
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "libero"
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]


def slugify(txt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", txt.lower()).strip("_")


def main(data_dir: str, *, push_to_hub: bool = False, idx: int = 0, output_dir: str | None = None):
    lerobot_home = output_dir or str(LEROBOT_HOME)

    datasets: dict[str, LeRobotDataset] = {}

    def get_ds(task_slug: str) -> LeRobotDataset:
        if task_slug not in datasets:
            repo_id = f"{REPO_NAME}-{RAW_DATASET_NAMES[idx]}/{task_slug}"
            out_dir = pathlib.Path(lerobot_home) / repo_id
            datasets[task_slug] = LeRobotDataset.create(
                repo_id=repo_id,
                root=out_dir,
                robot_type="panda",
                fps=10,
                features={
                    "image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "wrist_image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "state": {
                        "dtype": "float32",
                        "shape": (8,),
                        "names": ["state"],
                    },
                    "actions": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["actions"],
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,
            )
        return datasets[task_slug]

    for raw_dataset_name in RAW_DATASET_NAMES[idx:idx + 1]:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")

        for episode in raw_dataset:
            dataset = None
            for step in episode["steps"].as_numpy_iterator():
                if dataset is None:
                    task_text = step["language_instruction"].decode()
                    task_slug = slugify(task_text)
                    dataset = get_ds(task_slug)

                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    if push_to_hub:
        for task_slug, dataset in datasets.items():
            dataset.push_to_hub(
                tags=["libero", "panda", "rlds"],
                private=False,
                push_videos=True,
                license="apache-2.0",
            )


if __name__ == "__main__":
    tyro.cli(main)
