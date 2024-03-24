import argparse
import datetime
import getpass
import json
import submitit
import copy
import functools
import pandas as pd
try:
    from virtual_fs import virtual_os as os
    from virtual_fs.virtual_io import open
except ImportError:
    import os
import subprocess
import ego4d.internal.human_pose.extract_camera_info as extract_camera_info


def add_arguments(parser):
    parser.add_argument("--config-name", default="georgiatech_covid_02_2")
    parser.add_argument(
        "--config_path", default="configs", help="Path to the config folder"
    )    
    parser.add_argument(
        "--take_names",
        default="georgiatech_covid_02_2",
        type=str,
        help="take names to run, concatenated by '+', "
        + "e.g., uniandes_dance_007_3+iiith_cooking_23+nus_covidtest_01",
    )
    parser.add_argument(
        "--base_work_dir",
        default=f"/checkpoint/{getpass.getuser()}/flow/ego4d/default",
        help="Base working directory",
    )
    parser.add_argument(
        "--batch_job_name",
        default=None,
        help="The name of the batch of the jobs",
    )
    parser.add_argument(
        "--entitlement",
        default="fair_gpu_pnb",
        help="entitlement for the flow job.",
    )
    parser.add_argument(
        "--partition",
        default="eht",
        help="partition for the devfair flow job, e.g., eht, learnaccel",
    )

    parser.add_argument("--gpu-type", default="volta32gb", choices=["volta32gb"])
    parser.add_argument("--name", default="default", help="Flow name")
    parser.add_argument("--num_machines", default=1, type=int)
    parser.add_argument(
        "--num_cpus", default=40, type=int, help="Number of CPUs **per machine**"
    )
    parser.add_argument(
        "--num_gpus", default=8, type=int, help="Number of GPUs **per machine**"
    )
    parser.add_argument("--ram-gb", default=200, type=int)
    parser.add_argument("--retry", default=1, type=int, help="Number of retries")
    parser.add_argument(
        "--run_type",
        default="submitit",
        choices=["flow_canary", "flow_local", "local", "submitit"],
        help="Whether launch job in submitit, flow-canary, flow-local or run local",
    )
    parser.add_argument(
        "--secure_group",
        default="fair_research_and_engineering",
        help="Secure group for job",
    )

    parser.add_argument(
        "--work_dir",
        default=None,
        help="Work directory",
    )
    parser.add_argument(
        "opts",
        help="additional options",
        default=None,
        nargs=argparse.REMAINDER,
    )


def create_executor(args, log_dir):    
    executor = submitit.AutoExecutor(folder=log_dir+"/%j")    
    executor.update_parameters(
        gpus_per_node=args.num_gpus,        
        cpus_per_task=10*args.num_gpus,
        nodes=args.num_machines,
        timeout_min=3 * 24 * 60,
        name=args.name,        
        slurm_partition=args.partition,
        slurm_constraint=args.gpu_type,
    )  
    executor.update_parameters(array_parallelism=2)
    return executor


def main():
    parser = argparse.ArgumentParser()    
    add_arguments(parser)
    args = parser.parse_args()   
    print(args)

    release_takes = pd.read_csv('/private/home/suyogjain/egoexo/jan24/release_take_metadata.csv')
    take_names = list(release_takes['take_name'])
    take_names = [x for x in take_names if 'bouldering' not in x]
    print(len(take_names))

    log_dir = os.path.join(args.base_work_dir, args.batch_job_name)    
    executor = create_executor(args, log_dir)
    print(executor)
    
    func = functools.partial(extract_camera_info.run_pipeline, args)
    jobs = executor.map_array(func, take_names)
    print(f"Jobs: {jobs}")   
    

if __name__ == "__main__":
    main()
