import argparse
import datetime
import getpass
import json
import submitit
import copy
import functools

try:
    from virtual_fs import virtual_os as os
    from virtual_fs.virtual_io import open
except ImportError:
    import os
import subprocess
import ego4d.internal.human_pose.extract_camera_info as extract_camera_info


def add_arguments(parser):
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
        "--num_gpus", default=2, type=int, help="Number of GPUs **per machine**"
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


def create_work_dir_list(args):
    if args.batch_job_name is None:
        # use date as the default batch job name
        args.batch_job_name = datetime.date.today().strftime("%Y%m%d")

    args.work_dir_list = []

    if args.work_dir == "None":
        args.work_dir = None

    if args.work_dir is None:
        max_id = 100
        batch_job_dir = os.path.join(args.base_work_dir, args.batch_job_name)
        for job in args.job_list:
            id = 1
            while id < max_id:
                work_dir = os.path.join(batch_job_dir, f"{job}-{id}")
                if os.path.exists(work_dir):
                    id += 1
                else:
                    os.makedirs(work_dir, exist_ok=False)
                    args.work_dir_list.append(work_dir)
                    break
            assert (
                id < max_id
            ), f"Could not find an available id for {batch_job_dir}/{job}"
    else:
        assert (
            len(args.job_list) == 1
        ), "Cannot use work_dir when there are more than 1 job"
        assert os.path.exists(args.work_dir), (
            f"{args.work_dir} does not exist. If you are not resuming "
            + "a previous training job, please set work_dir to None"
        )
        args.work_dir_list.append(args.work_dir)
        print(
            f"[Warning] The job will be resumed under existing work_dir {args.work_dir}"
        )

    args.forwarded_opts = ""

def create_executor(args, log_dir):    
    executor = submitit.AutoExecutor(folder=log_dir+"%j")    
    executor.update_parameters(
        gpus_per_node=args.num_gpus,        
        cpus_per_task=80,
        nodes=args.num_machines,
        timeout_min=3 * 24 * 60,
        name=args.name,        
        slurm_partition=args.partition,
        slurm_constraint=args.gpu_type,
    )  
    executor.update_parameters(array_parallelism=2)
    return executor


def run_task(task):
    parser = argparse.ArgumentParser()    
    add_arguments(parser)
    args = parser.parse_args()   
    print(args)

    '''
    log_dir = os.path.join(args.base_work_dir, args.batch_job_name)    
    executor = create_executor(args, log_dir)

    take_name_list = args.take_names.split("+") 
    print(take_name_list)
    print(task)
    func = functools.partial(task, args)
    print(func)
    '''

    #jobs = executor.map_array(functools.partial(task, args=args), take_name_list)
    #print(f"Jobs: {jobs}")   
    
    #print("Args: {}".format(args))
    #jobs = executor.map_array(task, args, task_name)  # just a list of jobs


    '''
    for job_id in range(args.job_num):
        task.config_single_job(args, job_id)
        launch_job_array(task, args)
    '''