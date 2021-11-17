# Copyright 2021 Alex Yu

import torch
import torch.nn.functional as F
import numpy as np
import random
from multiprocessing import Process, Queue
import os
from os import path, listdir
import argparse
import json
import subprocess
import sys
from typing import List, Dict
import itertools
from warnings import warn
from datetime import datetime
import numpy as np
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("task_json", type=str)
parser.add_argument("--gpus", "-g", type=str, required=True,
                            help="space delimited GPU id list (global id in nvidia-smi, "
                                 "not considering CUDA_VISIBLE_DEVICES)")
parser.add_argument('--eval', action='store_true', default=False,
                   help='evaluation mode (run the render_imgs script)')
parser.add_argument('--render', action='store_true', default=False,
                   help='also run render_imgs.py with --render_path to render a rotating trajectory (forward-facing case)')
args = parser.parse_args()

PSNR_FILE_NAME = 'test_psnr.txt'

def run_exp(env, eval_mode:bool, enable_render:bool, train_dir, data_dir, config, flags, eval_flags, common_flags):
    opt_base_cmd = [ "python", "opt.py", "--tune_mode" ]

    if not eval_mode:
        opt_base_cmd += ["--tune_nosave"]
    opt_base_cmd += [
        "-t", train_dir,
        data_dir
    ]
    if config != '':
        opt_base_cmd += ['-c', config]
    log_file_path = path.join(train_dir, 'log')
    psnr_file_path = path.join(train_dir, PSNR_FILE_NAME)
    ckpt_path = path.join(train_dir, 'ckpt.npz')
    if path.isfile(psnr_file_path):
        print('! SKIP', train_dir)
        return
    print('********************************************')
    if eval_mode:
        print('EVAL MODE')

    if eval_mode and path.isfile(ckpt_path):
        print('! SKIP training because ckpt exists', ckpt_path)
        opt_ret = ""  # Silence
    else:
        print('! RUN opt.py -t', train_dir)
        opt_cmd = ' '.join(opt_base_cmd + flags + common_flags)
        print(opt_cmd)
        try:
            opt_ret = subprocess.check_output(opt_cmd, shell=True, env=env).decode(
                    sys.stdout.encoding)
        except subprocess.CalledProcessError:
            print('Error occurred while running OPT for exp', train_dir, 'on', env["CUDA_VISIBLE_DEVICES"])
            return
        with open(log_file_path, 'w') as f:
            f.write(opt_ret)

    if eval_mode:
        eval_base_cmd = [
            "python", "render_imgs.py",
            ckpt_path,
            data_dir
        ]
        if config != '':
            eval_base_cmd += ['-c', config]
        psnr_file_path = path.join(train_dir, 'test_renders', 'psnr.txt')
        if not path.exists(psnr_file_path):
            eval_cmd = ' '.join(eval_base_cmd + eval_flags + common_flags)
            print('! RUN render_imgs.py', ckpt_path)
            print(eval_cmd)
            try:
                eval_ret = subprocess.check_output(eval_cmd, shell=True, env=env).decode(
                        sys.stdout.encoding)
            except subprocess.CalledProcessError:
                print('Error occurred while running EVAL for exp', train_dir, 'on', env["CUDA_VISIBLE_DEVICES"])
                return
        else:
            print('! SKIP eval because psnr.txt exists', psnr_file_path)

        if enable_render:
            eval_base_cmd += ['--render_path']
            render_cmd = ' '.join(eval_base_cmd + eval_flags + common_flags)
            try:
                render_ret = subprocess.check_output(render_cmd, shell=True, env=env).decode(
                        sys.stdout.encoding)
            except subprocess.CalledProcessError:
                print('Error occurred while running RENDER for exp', train_dir, 'on', env["CUDA_VISIBLE_DEVICES"])
                return
    else:
        test_stats = [eval(x.split('eval stats:')[-1].strip())
                      for x in opt_ret.split('\n') if
                      x.startswith('eval stats: ')]
        if len(test_stats) == 0:
            print('note: invalid config or crash')
            final_test_psnr = 0.0
        else:
            test_psnrs = [stats['psnr'] for stats in test_stats if 'psnr' in stats.keys()]
            print('final psnrs', test_psnrs[-5:])
            final_test_psnr = test_psnrs[-1]
        with open(psnr_file_path, 'w') as f:
            f.write(str(final_test_psnr))

def process_main(device, eval_mode:bool, enable_render:bool, queue):
    # Set CUDA_VISIBLE_DEVICES programmatically
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    while True:
        task = queue.get()
        if len(task) == 0:
            break
        run_exp(env, eval_mode, enable_render, **task)

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

def create_prodvars(variables, noise_stds={}):
    """
    Create a dict for each setting of variable values
    (product across lists)
    """

    def auto_list(x):
        if isinstance(x, list):
            return x
        elif isinstance(x, dict) or isinstance(x, set):
            return [x]
        elif isinstance(x, str):
            return eval(x)
        else:
            raise NotImplementedError('variable value must be list of values, or str generator')

    variables = {varname:auto_list(variables[varname]) for varname in variables}
    print('variables (prod)', variables)
    varnames = list(variables.keys())
    noise_stds = np.array([noise_stds.get(varname, 0.0) for varname in varnames])
    variables = [[(i, val) for val in variables[varname]] for i, varname in enumerate(varnames)]
    prodvars = list(itertools.product(*variables))
    noise_vals = np.random.randn(len(prodvars), len(varnames)) * noise_stds
    prodvars = [{varnames[i]:((val + n) if n != 0.0 else val) for (i, val), n in zip(sample, noise_vals_samp)} for sample, noise_vals_samp in zip(prodvars, noise_vals)]
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
    base_eval_flags = tasks_file.get('base_eval_flags', [])
    base_common_flags = tasks_file.get('base_common_flags', [])
    default_config = tasks_file.get('config', '')

    if 'eval' in tasks_file:
        args.eval = tasks_file['eval']
        print('Eval mode?', args.eval)
    if 'render' in tasks_file:
        args.render = tasks_file['render']
        print('Render traj?', args.render)
    pqueue = Queue()

    leaderboard_path = path.join(train_root, 'results.txt' if args.eval else 'leaderboard.txt')
    print('Leaderboard path:', leaderboard_path)

    variables : Dict = tasks_file.get('variables', {})
    noises : Dict = tasks_file.get('noises', {})
    assert isinstance(variables, dict), 'var must be dict'

    prodvars : List[Dict] = create_prodvars(variables, noises)
    del variables

    for task_templ in all_tasks_templ:
        for variables in prodvars:
            task : Dict = recursive_replace(task_templ, variables)
            task['train_dir'] = path.join(train_root, task['train_dir'])  # Required
            task['data_dir'] = path.join(data_root, task.get('data_dir', '')).rstrip('/')
            task['flags'] = task.get('flags', []) + base_flags
            task['eval_flags'] = task.get('eval_flags', []) + base_eval_flags
            task['common_flags'] = task.get('common_flags', []) + base_common_flags
            task['config'] = task.get('config', default_config)
            os.makedirs(task['train_dir'], exist_ok=True)
            # santity check
            assert path.exists(task['train_dir']), task['train_dir'] + ' does not exist'
            assert path.exists(task['data_dir']), task['data_dir'] + ' does not exist'
            all_tasks.append(task)
    task = None
    # Shuffle the tasks
    if not args.eval:
        random.shuffle(all_tasks)

    for task in all_tasks:
        pqueue.put(task)

    args.gpus = list(map(int, args.gpus.split()))
    print('GPUS:', args.gpus)

    for _ in args.gpus:
        pqueue.put({})

    all_procs = []
    for i, gpu in enumerate(args.gpus):
        process = Process(target=process_main, args=(gpu, args.eval, args.render, pqueue))
        process.daemon = True
        process.start()
        all_procs.append(process)

    for i, gpu in enumerate(args.gpus):
        all_procs[i].join()

    if args.eval:
        print('Done')
        with open(leaderboard_path, 'w') as leaderboard_file:
            lines = [f'dir\tPSNR\tSSIM\tLPIPS\nminutes\n']
            all_tasks = sorted(all_tasks, key=lambda task:task['train_dir'])
            all_psnr = []
            all_ssim = []
            all_lpips = []
            all_times = []
            for task in all_tasks:
                train_dir = task['train_dir']
                psnr_file_path = path.join(train_dir, 'test_renders', 'psnr.txt')
                ssim_file_path = path.join(train_dir, 'test_renders', 'ssim.txt')
                lpips_file_path = path.join(train_dir, 'test_renders', 'lpips.txt')
                time_file_path = path.join(train_dir, 'time_mins.txt')

                if path.isfile(psnr_file_path):
                    with open(psnr_file_path, 'r') as f:
                        psnr = float(f.read())
                    all_psnr.append(psnr)
                    psnr_txt = f'{psnr:.10f}'
                else:
                    psnr_txt = 'ERR'
                if path.isfile(ssim_file_path):
                    with open(ssim_file_path, 'r') as f:
                        ssim = float(f.read())
                    all_ssim.append(ssim)
                    ssim_txt = f'{ssim:.10f}'
                else:
                    ssim_txt = 'ERR'
                if path.isfile(lpips_file_path):
                    with open(lpips_file_path, 'r') as f:
                        lpips = float(f.read())
                    all_lpips.append(lpips)
                    lpips_txt = f'{lpips:.10f}'
                else:
                    lpips_txt = 'ERR'
                if path.isfile(time_file_path):
                    with open(time_file_path, 'r') as f:
                        time_mins = float(f.read())
                    all_times.append(time_mins)
                    time_txt = f'{time_mins:.10f}'
                else:
                    time_txt = 'ERR'
                line = f'{path.basename(train_dir.rstrip("/"))}\t{psnr_txt}\t{ssim_txt}\t{lpips_txt}\t{time_txt}\n'
                lines.append(line)
            lines.append('---------\n')
            if len(all_psnr):
                lines.append('Average PSNR: ' + str(sum(all_psnr) / len(all_psnr)) + '\n')
            if len(all_ssim):
                lines.append('Average SSIM: ' + str(sum(all_ssim) / len(all_ssim)) + '\n')
            if len(all_lpips):
                lines.append('Average LPIPS: ' + str(sum(all_lpips) / len(all_lpips)) + '\n')
            if len(all_times):
                lines.append('Average Time (mins): ' + str(sum(all_times) / len(all_times)) + '\n')
            leaderboard_file.writelines(lines)

    else:
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

