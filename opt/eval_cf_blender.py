# Script modified from https://github.com/jzhangbs/DTUeval-python

from genericpath import isdir
import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import json
from util import config_util

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='',
                        help='path to predicted pts')
    parser.add_argument('--gt_path', default='',
                        help='path to extracted ground turth pts')

    parser.add_argument('--downsample_density', type=float, default=0.001)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--visualize_threshold', type=float, default=0.1)
    parser.add_argument('--out_dir', type=str, default='./')
    # parser.add_argument('--del_ckpt', action='store_true', default=False)
    parser.add_argument('--no_pts_save', action='store_true', default=False)
    parser.add_argument('--log_tune_hparam_config_path', type=str, default=None,
                       help='Log hyperparamters being tuned to tensorboard based on givn config.json path')
    args = parser.parse_args()

    thresh = args.downsample_density
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    summary_writer = SummaryWriter(f'{os.path.dirname(args.input_path)}/../')

    if os.path.isdir(args.input_path):
        data_pcd = np.load(f'{args.input_path}/pts.npy')
    else:
        data_pcd = np.load(args.input_path)


    if os.path.isdir(args.gt_path):
        stl = np.load(f'{args.gt_path}/shape.npy')
    else:
        stl = np.load(args.gt_path)



    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_pcd, n_neighbors=1, return_distance=True)
    max_dist = args.max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    nn_engine.fit(data_pcd)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    vis_dist = args.visualize_threshold
    R = np.array([[1,0,0]], dtype=np.float64)
    G = np.array([[0,1,0]], dtype=np.float64)
    B = np.array([[0,0,1]], dtype=np.float64)
    W = np.array([[1,1,1]], dtype=np.float64)
    data_color = np.tile(B, (data_pcd.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    data_color = R * data_alpha + W * (1-data_alpha)
    data_color[dist_d2s[:,0] >= max_dist] = G
    # data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
    # data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
    stl_color = np.tile(B, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color= R * stl_alpha + W * (1-stl_alpha)
    stl_color[dist_s2d[:,0] >= max_dist] = G
    # stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
    # stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
    over_all = (mean_d2s + mean_s2d) / 2
    print(mean_d2s, mean_s2d, over_all)
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        if not args.no_pts_save:
            write_vis_pcd(f'{args.out_dir}/vis_d2s.ply', data_pcd, data_color)
            write_vis_pcd(f'{args.out_dir}/vis_s2d.ply', stl, stl_color)


        with open(f'{args.out_dir}/cf.txt', 'w') as f:
            f.write(f'Mean d2s: {mean_d2s}\n')
            f.write(f'Mean s2d: {mean_s2d}\n')
            f.write(f'Over all: {over_all}\n')


    # log hparams for tuning tasks
    if args.log_tune_hparam_config_path is not None:
        train_args = config_util.setup_train_conf(return_parpser=True).parse_known_args(
            args=['-c', f'{os.path.dirname(args.input_path)}/../config.yaml',
            '--data_dir', 'foo']
            )[0]
        with open(args.log_tune_hparam_config_path, 'r') as f:
            tune_conf = json.load(f)
        hparams = {}
        for hp in tune_conf['params']:
            arg = hp['text'].split('=')[0].strip()
            value = getattr(train_args, arg)
            hparams[arg] = value
        
        metrics = {
            'Chamfer/d2s': mean_d2s,
            'Chamfer/s2d': mean_s2d,
            'Chamfer/mean': over_all,
        }
        summary_writer.add_hparams(hparams, metrics, run_name=os.path.realpath(f'{os.path.dirname(args.input_path)}/../'))
        summary_writer.flush()
    else:

        summary_writer.add_scalar('Chamfer/d2s', mean_d2s, global_step=0)
        summary_writer.add_scalar('Chamfer/s2d', mean_s2d, global_step=0)
        summary_writer.add_scalar('Chamfer/mean', over_all, global_step=0)
        summary_writer.flush()
    summary_writer.close()
