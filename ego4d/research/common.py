import math
from dataclasses import dataclass
from typing import Any, List

import submitit


@dataclass
class SlurmConfig:
    slurm_log_folder: str
    timeout_min: int
    constraint: str
    slurm_partition: str
    slurm_array_parallelism: int
    gpus_per_node: int
    cpus_per_task: int
    run_locally: bool = False


def batch_it(things: List[Any], batch_size: int) -> List[List[Any]]:
    num_batches: int = math.ceil(len(things) / batch_size)

    result = []
    for i in range(num_batches):
        result.append(things[i * batch_size : (i + 1) * batch_size])
    return result


def create_executor(config: SlurmConfig, num_batches: int):
    if config.run_locally:
        executor = submitit.LocalExecutor(folder=config.slurm_log_folder)
    else:
        executor = submitit.AutoExecutor(folder=config.slurm_log_folder)

    executor.update_parameters(
        timeout_min=config.timeout_min,
        slurm_constraint=config.constraint,
        slurm_partition=config.slurm_partition,
        slurm_array_parallelism=min(config.slurm_array_parallelism, num_batches),
        gpus_per_node=config.gpus_per_node,
        cpus_per_task=config.cpus_per_task,
    )
    return executor
