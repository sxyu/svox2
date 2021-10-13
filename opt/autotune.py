# Copyright 2021 Alex Yu

import torch
import torch.nn.functional as F
import numpy as np
import random
from multiprocessing import Process, Queue
import os
from os import path
import argparse
import json
import subprocess
import sys
from typing import List, Dict
import itertools
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("task_json", type=str)
parser.add_argument("--gpus", "-g", type=str, required=True,
                            help="space delimited GPU id list (global id in nvidia-smi, "
                                 "not considering CUDA_VISIBLE_DEVICES)")
args = parser.parse_args()

PSNR_FILE_NAME = 'test_psnr.txt'

def run_exp(env, train_dir, data_dir, flags):
    opt_base_cmd = [
        "python", "-u", "opt.py", "--tune_mode",
        "-t", train_dir,
        data_dir
    ]
    log_file_path = path.join(train_dir, 'log')
    psnr_file_path = path.join(train_dir, PSNR_FILE_NAME)
    print('********************************************')
    print('! RUN opt.py -t', train_dir)
    opt_cmd = ' '.join(opt_base_cmd + flags)
    print(opt_cmd)
    opt_ret = subprocess.check_output(opt_cmd, shell=True, env=env).decode(
            sys.stdout.encoding)
    with open(log_file_path, 'w') as f:
        f.write(opt_ret)
    test_stats = [eval(x.split('eval stats:')[-1].strip())
                  for x in opt_ret.split('\n') if
                  x.startswith('eval stats: ')]
    test_psnrs = [stats['psnr'] for stats in test_stats if 'psnr' in stats.keys()]
    print('final psnrs', test_psnrs[-5:])
    final_test_psnr = test_psnrs[-1]
    with open(psnr_file_path, 'w') as f:
        f.write(str(final_test_psnr))

def process_main(device, queue):
    # Set CUDA_VISIBLE_DEVICES programmatically
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    while True:
        task = queue.get()
        if len(task) == 0:
            break
        run_exp(env, **task)

# Variable value list generation helpers
def lin(start, stop, num):
    return np.linspace(start, stop, num).tolist()

def randlin(start, stop, num):
    lst = np.linspace(start, stop, num + 1)[:-1]
    lst += np.random.uniform(low=0.0, high=(lst[1] - lst[0]), size=lst.shape)
    return lst.tolist()

def loglin(start, stop, num):
    return np.exp(np.linspace(np.log(start), np.log(stop), num)).tolist()

def randloglin(start, stop, num):
    lst = np.linspace(np.log(start), np.log(stop), num + 1)[:-1]
    lst += np.random.uniform(low=0.0, high=(lst[1] - lst[0]), size=lst.shape)
    return np.exp(lst).tolist()
# End variable value list generation helpers

def create_prodvars(variables):
    """
    Create a dict for each setting of variable values
    (product across lists)
    """

    def auto_list(x):
        if isinstance(x, list):
            return x
        elif isinstance(x, str):
            return eval(x)
        else:
            raise NotImplementedError('variable value must be list of values, or str generator')

    variables = {varname:auto_list(variables[varname]) for varname in variables}
    print('variables (prod)', variables)
    variables = [[(varname, val) for val in variables[varname]] for varname in variables]
    prodvars = itertools.product(*variables)
    prodvars = [{varname:val for varname, val in sample} for sample in prodvars]
    return prodvars


def recursive_replace(data, variables):
    if isinstance(data, str):
        return data.format(**variables)
    elif isinstance(data, list):
        return [recursive_replace(d, variables) for d in data]
    elif isinstance(data, dict):
        return {k:recursive_replace(data[k], variables) for k in data.keys()}
    else:
        return data


if __name__ == '__main__':
    with open(args.task_json, 'r') as f:
        tasks_file = json.load(f)
    assert isinstance(tasks_file, dict), 'Root of json must be dict'
    all_tasks_templ = tasks_file.get('tasks', [])
    all_tasks = []
    data_root = path.expanduser(tasks_file['data_root'])  # Required
    train_root = path.expanduser(tasks_file['train_root'])  # Required
    base_flags = tasks_file.get('base_flags', [])
    pqueue = Queue()

    leaderboard_path = path.join(train_root, 'leaderboard.txt')
    print('Leaderboard path:', leaderboard_path)

    variables : Dict = tasks_file.get('variables', {})
    assert isinstance(variables, dict), 'var must be dict'

    prodvars : List[Dict] = create_prodvars(variables)
    del variables

    for task_templ in all_tasks_templ:
        for variables in prodvars:
            task : Dict = recursive_replace(task_templ, variables)
            task['train_dir'] = path.join(train_root, task['train_dir'])  # Required
            task['data_dir'] = path.join(data_root, task.get('data_dir', '')).rstrip('/')
            task['flags'] = task.get('flags', []) + base_flags
            os.makedirs(task['train_dir'], exist_ok=True)
            # santity check
            assert path.exists(task['train_dir']), task['train_dir'] + ' does not exist'
            assert path.exists(task['data_dir']), task['data_dir'] + ' does not exist'
            all_tasks.append(task)
    task = None
    # Shuffle the tasks
    random.shuffle(all_tasks)

    for task in all_tasks:
        pqueue.put(task)

    args.gpus = list(map(int, args.gpus.split()))
    print('GPUS:', args.gpus)

    for _ in args.gpus:
        pqueue.put({})

    all_procs = []
    for i, gpu in enumerate(args.gpus):
        process = Process(target=process_main, args=(gpu, pqueue))
        process.daemon = True
        process.start()
        all_procs.append(process)

    for i, gpu in enumerate(args.gpus):
        all_procs[i].join()


    with open(leaderboard_path, 'w') as leaderboard_file:
        exps = []
        for task in all_tasks:
            train_dir = task['train_dir']
            psnr_file_path = path.join(train_dir, PSNR_FILE_NAME)

            with open(psnr_file_path, 'r') as f:
                test_psnr = float(f.read())
                print(train_dir, test_psnr)
                exps.append((test_psnr, train_dir))
        exps = sorted(exps, key = lambda x: -x[0])
        lines = [f'{psnr:.10f}\t{train_dir}\n' for psnr, train_dir in exps]
        leaderboard_file.writelines(lines)
    print('Wrote', leaderboard_path)

