import glob
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import tyro
import wandb


@dataclass
class Args:
    blender_path: str
    """Path to the blender executable"""

    workers_per_gpu: int
    """number of workers per gpu"""

    input_models_path: str
    """Path to a json file containing a list of 3D object files"""

    num_gpus: int = -1
    """number of gpus to use. -1 means all available gpus"""

    output_dir: str = "output"
    """output directory"""


def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        # Perform some operation on the item
        print(item, gpu)
        output_dir = args.output_dir
        blender_path = args.blender_path
        obj_uid = item.split("/")[-1].split(".")[0]
        if not os.path.exists(f"{output_dir}/{obj_uid}"):
            command = (
                f"export DISPLAY=:0.{gpu} &&"
                f" {blender_path} -b -P blender/blender_script.py --"
                f" --object_path {item}"
                f" --output_dir {output_dir} >> tmp.out"
            )
            subprocess.run(command, shell=True)
        else:
            print(f"Skipping {item} as it already exists")

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i)
            )
            process.daemon = True
            process.start()

    # Add items to the queue
    with open(args.input_models_path, "r") as f:
        model_paths = json.load(f)

    random.shuffle(model_paths)
    for item in model_paths:
        queue.put(item)

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)
