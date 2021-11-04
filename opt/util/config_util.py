import torch
import argparse
from util.dataset import datasets
import yaml


def define_common_args(parser : argparse.ArgumentParser):
    parser.add_argument('data_dir', type=str)

    parser.add_argument('--config', '-c',
                         type=str,
                         default=None,
                         help="Config yaml file (will override args)")

    parser.add_argument_group("Data loading")
    parser.add_argument('--dataset_type',
                         choices=list(datasets.keys()) + ["auto"],
                         default="auto",
                         help="Dataset type (specify type or use auto)")
    parser.add_argument('--scene_scale',
                         type=float,
                         default=None,
                         help="Global scene scaling (or use dataset default)")
    parser.add_argument('--scale',
                         type=float,
                         default=None,
                         help="Image scale, e.g. 0.5 for half size (or use dataset default)")
    parser.add_argument('--white_bkgd',
                         type=bool,
                         default=True,
                         help="Whether to use white background (ignored in some datasets)")
    parser.add_argument('--llffhold',
                         type=int,
                         default=8,
                         help="LLFF holdout every")
    parser.add_argument('--normalize_by_bbox',
                         type=bool,
                         default=True,
                         help="Normalize by bounding box in bbox.txt, if available (NSVF dataset only)")
    parser.add_argument('--data_bbox_scale',
                         type=float,
                         default=1.2,
                         help="Data bbox scaling (NSVF dataset only)")

    parser.add_argument_group("Render options")
    parser.add_argument('--step_size',
                         type=float,
                         default=0.5,
                         help="Render step size (in voxel size units)")
    parser.add_argument('--sigma_thresh',
                         type=float,
                         default=1e-8,
                         help="Skips voxels with sigma < this")
    parser.add_argument('--stop_thresh',
                         type=float,
                         default=1e-7,
                         help="Ray march stopping threshold")
    parser.add_argument('--background_brightness',
                         type=float,
                         default=1.0,
                         help="Brightness of the infinite background")
    parser.add_argument('--renderer_backend',
                         choices=['cuvol'],
                         default='cuvol',
                         help="Renderer backend")
    parser.add_argument('--background_msi_scale',
                         type=float,
                         default=1.0,
                         help="For BG model, distance of nearest MPI layer to origin (relative to voxel grid radius)")


def build_data_options(args):
    return {
        'dataset_type': args.dataset_type,
        'scene_scale': args.scene_scale,
        'scale': args.scale,
        'white_bkgd': args.white_bkgd,
        'hold_every': args.llffhold,
        'normalize_by_bbox': args.normalize_by_bbox,
        'data_bbox_scale': args.data_bbox_scale
    }

def maybe_merge_config_file(args):
    """
    Load yaml config file if specified and merge the arguments
    """
    if args.config is not None:
        with open(args.config, "r") as config_file:
            configs = yaml.load(config_file, Loader=yaml.FullLoader)
        invalid_args = list(set(configs.keys()) - set(dir(args)))
        if invalid_args:
            raise ValueError(f"Invalid args {invalid_args} in {args.config}.")
        args.__dict__.update(configs)
