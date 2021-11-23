import torch
import argparse
from util.dataset import datasets
import json


def define_common_args(parser : argparse.ArgumentParser):
    parser.add_argument('data_dir', type=str)

    parser.add_argument('--config', '-c',
                         type=str,
                         default=None,
                         help="Config yaml file (will override args)")

    group = parser.add_argument_group("Data loading")
    group.add_argument('--dataset_type',
                         choices=list(datasets.keys()) + ["auto"],
                         default="auto",
                         help="Dataset type (specify type or use auto)")
    group.add_argument('--scene_scale',
                         type=float,
                         default=None,
                         help="Global scene scaling (or use dataset default)")
    group.add_argument('--scale',
                         type=float,
                         default=None,
                         help="Image scale, e.g. 0.5 for half resolution (or use dataset default)")
    group.add_argument('--seq_id',
                         type=int,
                         default=1000,
                         help="Sequence ID (for CO3D only)")
    group.add_argument('--epoch_size',
                         type=int,
                         default=12800,
                         help="Pseudo-epoch size in term of batches (to be consistent across datasets)")
    group.add_argument('--white_bkgd',
                         type=bool,
                         default=True,
                         help="Whether to use white background (ignored in some datasets)")
    group.add_argument('--llffhold',
                         type=int,
                         default=8,
                         help="LLFF holdout every")
    group.add_argument('--normalize_by_bbox',
                         type=bool,
                         default=False,
                         help="Normalize by bounding box in bbox.txt, if available (NSVF dataset only); precedes normalize_by_camera")
    group.add_argument('--data_bbox_scale',
                         type=float,
                         default=1.2,
                         help="Data bbox scaling (NSVF dataset only)")
    group.add_argument('--cam_scale_factor',
                         type=float,
                         default=0.95,
                         help="Camera autoscale factor (NSVF/CO3D dataset only)")
    group.add_argument('--normalize_by_camera',
                         type=bool,
                         default=True,
                         help="Normalize using cameras, assuming a 360 capture (NSVF dataset only); only used if not normalize_by_bbox")
    group.add_argument('--perm', action='store_true', default=False,
                         help='sample by permutation of rays (true epoch) instead of '
                              'uniformly random rays')

    group = parser.add_argument_group("Render options")
    group.add_argument('--step_size',
                         type=float,
                         default=0.5,
                         help="Render step size (in voxel size units)")
    group.add_argument('--sigma_thresh',
                         type=float,
                         default=1e-8,
                         help="Skips voxels with sigma < this")
    group.add_argument('--stop_thresh',
                         type=float,
                         default=1e-7,
                         help="Ray march stopping threshold")
    group.add_argument('--background_brightness',
                         type=float,
                         default=1.0,
                         help="Brightness of the infinite background")
    group.add_argument('--renderer_backend', '-B',
                         choices=['cuvol', 'svox1', 'nvol'],
                         default='cuvol',
                         help="Renderer backend")
    group.add_argument('--random_sigma_std',
                         type=float,
                         default=0.0,
                         help="Random Gaussian std to add to density values (only if enable_random)")
    group.add_argument('--random_sigma_std_background',
                         type=float,
                         default=0.0,
                         help="Random Gaussian std to add to density values for BG (only if enable_random)")
    group.add_argument('--near_clip',
                         type=float,
                         default=0.00,
                         help="Near clip distance (in world space distance units, only for FG)")
    group.add_argument('--use_spheric_clip',
                         action='store_true',
                         default=False,
                         help="Use spheric ray clipping instead of voxel grid AABB "
                              "(only for FG; changes near_clip to mean 1-near_intersection_radius; "
                              "far intersection is always at radius 1)")
    group.add_argument('--enable_random',
                         action='store_true',
                         default=False,
                         help="Random Gaussian std to add to density values")
    group.add_argument('--last_sample_opaque',
                         action='store_true',
                         default=False,
                         help="Last sample has +1e9 density (used for LLFF)")


def build_data_options(args):
    """
    Arguments to pass as kwargs to the dataset constructor
    """
    return {
        'dataset_type': args.dataset_type,
        'seq_id': args.seq_id,
        'epoch_size': args.epoch_size * args.__dict__.get('batch_size', 5000),
        'scene_scale': args.scene_scale,
        'scale': args.scale,
        'white_bkgd': args.white_bkgd,
        'hold_every': args.llffhold,
        'normalize_by_bbox': args.normalize_by_bbox,
        'data_bbox_scale': args.data_bbox_scale,
        'cam_scale_factor': args.cam_scale_factor,
        'normalize_by_camera': args.normalize_by_camera,
        'permutation': args.perm
    }

def maybe_merge_config_file(args, allow_invalid=False):
    """
    Load json config file if specified and merge the arguments
    """
    if args.config is not None:
        with open(args.config, "r") as config_file:
            configs = json.load(config_file)
        invalid_args = list(set(configs.keys()) - set(dir(args)))
        if invalid_args and not allow_invalid:
            raise ValueError(f"Invalid args {invalid_args} in {args.config}.")
        args.__dict__.update(configs)

def setup_render_opts(opt, args):
    """
    Pass render arguments to the SparseGrid renderer options
    """
    opt.step_size = args.step_size
    opt.sigma_thresh = args.sigma_thresh
    opt.stop_thresh = args.stop_thresh
    opt.background_brightness = args.background_brightness
    opt.backend = args.renderer_backend
    opt.random_sigma_std = args.random_sigma_std
    opt.random_sigma_std_background = args.random_sigma_std_background
    opt.last_sample_opaque = args.last_sample_opaque
    opt.near_clip = args.near_clip
    opt.use_spheric_clip = args.use_spheric_clip
