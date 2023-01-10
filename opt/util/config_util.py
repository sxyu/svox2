import torch
import configargparse
from util.dataset import datasets
import json
import numpy as np


def define_common_args(parser : configargparse.ArgumentParser):
    parser.add_argument('data_dir', type=str)

    parser.add_argument('--config', '-c', 
                         is_config_file=True,
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
                         choices=['cuvol', 'svox1', 'nvol', 'surface', 'surf_trav'],
                         default='cuvol',
                         help="Renderer backend")
    group.add_argument('--surf_alpha_sigmoid_act',
                         action='store_true',
                         default=False,
                         help="Use exp activation for surf alpha")
    group.add_argument('--surface_type',
                        #  choices=['sdf', 'plane', 'udf', 'udf_alpha', 'udf_fake_sample'],
                         default=None,
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

    group.add_argument('--nokernel', action='store_true', default=False,
                        help='do not use cuda kernel to speed up training')

    group.add_argument('--force_alpha', action='store_true', default=False,
                        help='clamp alpha to be non-trivial during training to force surface learning')

    group.add_argument('--save_all_ckpt', action='store_true', default=False,
                        help='Save all ckpts during training')

    group.add_argument('--refresh_iter', 
                        type=int,
                        default=1,
                        help='do not use cuda kernel to speed up training')

    group.add_argument('--surf_fake_sample', 
                        action='store_true',
                        default=False,
                        help='Render with fake sample for surface')
    group.add_argument('--limited_fake_sample', 
                        action='store_true',
                        default=False,
                        help='Only apply fake sample if the voxel contains a surface')
    group.add_argument('--surf_fake_sample_min_vox_len', 
                         type=float,
                         default=0.,
                        help='minimum length for the ray in voxel to be considered for fake sampling')
    group.add_argument('--no_surf_grad_from_sh', 
                        action='store_true',
                        default=False,
                        help='Disable gradients flowing back to surface from sh')
    group.add_argument('--no_fake_sample_l_dist', 
                        action='store_true',
                        default=False)


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
    raise NotImplementedError('No Longer Used!')
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
    
    # convert alpha threshold to raw alpha values
    if args.renderer_backend in ['surface', 'surf_trav'] and args.surf_alpha_sigmoid_act:
        opt.sigma_thresh = np.log(args.sigma_thresh / (1. - args.sigma_thresh))
    else:
        opt.sigma_thresh = args.sigma_thresh

    opt.stop_thresh = args.stop_thresh
    opt.background_brightness = args.background_brightness
    opt.backend = args.renderer_backend
    opt.random_sigma_std = args.random_sigma_std
    opt.random_sigma_std_background = args.random_sigma_std_background
    opt.last_sample_opaque = args.last_sample_opaque
    opt.near_clip = args.near_clip
    opt.use_spheric_clip = args.use_spheric_clip
    opt.surf_fake_sample = args.surf_fake_sample
    opt.surf_fake_sample_min_vox_len = args.surf_fake_sample_min_vox_len
    opt.limited_fake_sample = args.limited_fake_sample
    opt.no_surf_grad_from_sh = args.no_surf_grad_from_sh
    opt.alpha_activation_type = 0 if args.surf_alpha_sigmoid_act else 1
    opt.fake_sample_l_dist = not args.no_fake_sample_l_dist


def setup_train_conf(return_parpser=False):
    # gin_configs = FLAGS.gin_configs

    # print('*** Loading Gin configs from: %s', str(gin_configs))
    parser = configargparse.ArgumentParser()
    define_common_args(parser)

    group = parser.add_argument_group("general")
    group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                        help='checkpoint and logging directory')

    group.add_argument('--reso',
                            # type=str,
                            nargs="+",
                            default=[[256, 256, 256], [512, 512, 512]],
                        help='List of grid resolution (will be evaled as json);'
                                'resamples to the next one every upsamp_every iters, then ' +
                                'stays at the last one; ' +
                                'should be a list where each item is a list of 3 ints or an int')
    group.add_argument('--upsamp_every', type=lambda x: int(float(x)), default=
                        3 * 12800,
                        help='upsample the grid every x iters')
    group.add_argument('--init_iters', type=lambda x: int(float(x)), default=
                        0,
                        help='do not upsample for first x iters')
    group.add_argument('--no_surface_init_iters', type=lambda x: int(float(x)), default=
                        0,
                        help='Do not use surface for first x iters, then init surface from learned alpha values')
    group.add_argument('--no_surface_init_debug_ckpt', action='store_true', default=False,
                        help='Save a ckpt for no surface init')
    group.add_argument('--surface_init_freeze', type=lambda x: int(float(x)), default=
                        0,
                        help='freeze surface for a few more iterations after density init')
    group.add_argument('--alpha_lv_sets', type=float, default=
                        0.1,
                        help='Value of alpha used to init surface')
    group.add_argument('--surf_init_alpha_rescale', type=float, default=None,
                        help='Rescale the raw values of alpha after surface init from nerf')
    group.add_argument('--surface_init_rescale', type=float, default=
                        0.1,
                        help='Rescale the raw values of surfaces')
    group.add_argument('--surface_init_reset_alpha',action='store_true', default=False,
                        help='Reset alpha value after no surface init')
    group.add_argument('--surf_init_reset_all',action='store_true', default=False,
                        help='Reset all alpha and sh values after surface init')
    # group.add_argument('--init_surface_with_alpha',action='store_true', default=False,
    #                     help='During no surface init, directly train with alpha instead of density')
    group.add_argument('--upsample_density_add', type=float, default=
                        0.0,
                        help='add the remaining density by this amount when upsampling')

    group.add_argument('--basis_type',
                        choices=['sh', '3d_texture', 'mlp'],
                        default='sh',
                        help='Basis function type')

    group.add_argument('--basis_reso', type=int, default=32,
                    help='basis grid resolution (only for learned texture)')
    group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')

    group.add_argument('--mlp_posenc_size', type=int, default=4, help='Positional encoding size if using MLP basis; 0 to disable')
    group.add_argument('--mlp_width', type=int, default=32, help='MLP width if using MLP basis')

    group.add_argument('--background_nlayers', type=int, default=0,#32,
                    help='Number of background layers (0=disable BG model)')
    group.add_argument('--background_reso', type=int, default=512, help='Background resolution')



    group = parser.add_argument_group("optimization")
    group.add_argument('--n_iters', type=lambda x: int(float(x)), default=10 * 12800, help='total number of iters to optimize for')
    group.add_argument('--batch_size', type=lambda x: int(float(x)), default=
                        5000,
                        #100000,
                        #  2000,
                    help='batch size')


    # TODO: make the lr higher near the end
    group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
    group.add_argument('--lr_sigma', type=float, default=3e1, help='SGD/rmsprop lr for sigma')
    group.add_argument('--lr_sigma_final', type=float, default=5e-2)
    group.add_argument('--lr_sigma_decay_steps', type=lambda x: int(float(x)), default=250000)
    group.add_argument('--lr_sigma_delay_steps', type=lambda x: int(float(x)), default=15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)

    group.add_argument('--alpha_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
    group.add_argument('--lr_alpha', type=float, default=3e1, help='SGD/rmsprop lr for alpha (surface optimization)')
    group.add_argument('--lr_alpha_final', type=float, default=5e-2)
    group.add_argument('--lr_alpha_fix_delay', type=lambda x: int(float(x)), default=0)
    group.add_argument('--lr_alpha_decay_steps', type=lambda x: int(float(x)), default=250000)
    group.add_argument('--lr_alpha_delay_steps', type=lambda x: int(float(x)), default=15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_alpha_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)


    group.add_argument('--surface_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Surface optimizer")
    group.add_argument('--lr_surface', type=float, default=3e1, help='SGD/rmsprop lr for surface')
    group.add_argument('--lr_surface_final', type=float, default=5e-2)
    group.add_argument('--lr_surf_fix_delay', type=lambda x: int(float(x)), default=0)
    group.add_argument('--lr_surface_decay_steps', type=lambda x: int(float(x)), default=250000)
    group.add_argument('--lr_surface_delay_steps', type=lambda x: int(float(x)), default=15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_surface_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)
    group.add_argument('--surf_grad_abs_max', type=float, default=None, help='Apply max gradient clipping on the surface grad')

    group.add_argument('--trainable_fake_sample_std', action='store_true', default=False, help='use trainable fake sample std')
    group.add_argument('--lr_fake_sample_std', type=float, default=1e-1, help='SGD/rmsprop lr for fake_sample_std')
    group.add_argument('--lr_fake_sample_std_final', type=float, default=5e-2)
    group.add_argument('--lr_fake_sample_std_decay_steps', type=lambda x: int(float(x)), default=250000)
    group.add_argument('--lr_fake_sample_std_delay_steps', type=lambda x: int(float(x)), default=15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_fake_sample_std_delay_mult', type=float, default=1e-2)
    group.add_argument('--lambda_fake_sample_std_l1', type=float, default=1e-2)
    group.add_argument('--lambda_fake_sample_std_l2', type=float, default=1e-2)

    group.add_argument('--fake_sample_std', type=float, default=1, help='std for fake samples')
    group.add_argument('--fake_sample_std_final', type=float, default=0.05)
    group.add_argument('--fake_sample_std_decay_steps', type=lambda x: int(float(x)), default=50000)
    group.add_argument('--fake_sample_std_delay', type=lambda x: int(float(x)), default=0)


    group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
    group.add_argument('--lr_sh', type=float, default=
                        1e-2,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_sh_final', type=float,
                        default=
                        5e-6
                        )
    group.add_argument('--lr_sh_decay_steps', type=lambda x: int(float(x)), default=250000)
    group.add_argument('--lr_sh_delay_steps', type=lambda x: int(float(x)), default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)

    group.add_argument('--lr_sh_surf', type=float, default=
                        1e-2,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_sh_surf_final', type=float,
                        default=
                        5e-6
                        )
    group.add_argument('--lr_sh_surf_decay_steps', type=lambda x: int(float(x)), default=250000)
    group.add_argument('--lr_sh_surf_delay_steps', type=lambda x: int(float(x)), default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sh_surf_delay_mult', type=float, default=1e-2)
    group.add_argument('--lr_sh_surf_fix_delay', type=lambda x: int(float(x)), default=0)

    group.add_argument('--lr_fg_begin_step', type=lambda x: int(float(x)), default=0, help="Foreground begins training at given step number")

    # BG LRs
    group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Background optimizer")
    group.add_argument('--lr_sigma_bg', type=float, default=3e0,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_sigma_bg_final', type=float, default=3e-3,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_sigma_bg_decay_steps', type=lambda x: int(float(x)), default=250000)
    group.add_argument('--lr_sigma_bg_delay_steps', type=lambda x: int(float(x)), default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sigma_bg_delay_mult', type=float, default=1e-2)

    group.add_argument('--lr_color_bg', type=float, default=1e-1,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_color_bg_final', type=float, default=5e-6,#1e-4,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_color_bg_decay_steps', type=lambda x: int(float(x)), default=250000)
    group.add_argument('--lr_color_bg_delay_steps', type=lambda x: int(float(x)), default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_color_bg_delay_mult', type=float, default=1e-2)
    # END BG LRs

    group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
    group.add_argument('--lr_basis', type=float, default=#2e6,
                        1e-6,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_basis_final', type=float,
                        default=
                        1e-6
                        )
    group.add_argument('--lr_basis_decay_steps', type=lambda x: int(float(x)), default=250000)
    group.add_argument('--lr_basis_delay_steps', type=lambda x: int(float(x)), default=0,#15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_basis_begin_step', type=lambda x: int(float(x)), default=0)#4 * 12800)
    group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)

    group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

    group.add_argument('--print_every', type=lambda x: int(float(x)), default=20, help='print every iterations')
    group.add_argument('--save_every', type=lambda x: int(float(x)), default=10000,
                    help='save every x iterations')
    group.add_argument('--eval_every', type=lambda x: int(float(x)), default=10000,
                    help='evaluate every x epochs')
    group.add_argument('--eval_every_iter', type=lambda x: int(float(x)), default=10000,
                    help='evaluate every x iterations')
    group.add_argument('--extract_mesh_every', type=lambda x: int(float(x)), default=100000,
                    help='extract mesh every x iterations')
    group.add_argument('--mesh_sigma_thresh', type=float, default=None,
                    help='extract mesh threshold')

    group.add_argument('--init_sigma', type=float,
                    default=0.1,
                    help='initialization sigma')
    group.add_argument('--init_sigma_bg', type=float,
                    default=0.1,
                    help='initialization sigma (for BG)')

    # Extra logging
    group.add_argument('--log_mse_image', action='store_true', default=False)
    group.add_argument('--log_depth_map', action='store_true', default=False)
    group.add_argument('--log_normal_map', action='store_true', default=False)
    group.add_argument('--log_alpha_map', action='store_true', default=False)
    group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
            help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")


    group = parser.add_argument_group("misc experiments")
    group.add_argument('--thresh_type',
                        choices=["weight", "sigma", "alpha"],
                        default="weight",
                    help='Upsample threshold type')
    group.add_argument('--weight_thresh', type=float,
                        default=0.0005 * 512,
                        #  default=0.025 * 512,
                    help='Upsample weight threshold; will be divided by resulting z-resolution')
    group.add_argument('--density_thresh', type=float,
                        default=5.0,
                    help='Upsample sigma threshold')
    group.add_argument('--alpha_upsample_thresh', type=float,
                        default=1e-8,
                    help='Upsample threshold for surface alpha')
    group.add_argument('--background_density_thresh', type=float,
                        default=1.0+1e-9,
                    help='Background sigma threshold for sparsification')
    group.add_argument('--max_grid_elements', type=lambda x: int(float(x)),
                        default=44_000_000,
                    help='Max items to store after upsampling '
                            '(the number here is given for 22GB memory)')
    group.add_argument('--surf_non_expand', action='store_true', default=False,
                    help='Do not expand to already turned-off voxel during upsampling')

    group.add_argument('--tune_mode', action='store_true', default=False,
                    help='hypertuning mode (do not save, for speed)')
    group.add_argument('--tune_nosave', action='store_true', default=False,
                    help='do not save any checkpoint even at the end')



    group = parser.add_argument_group("losses")
    # group.add_argument('--img_lambda_l2', type=float, default=1)
    # group.add_argument('--img_lambda_l1', type=float, default=0)
    group.add_argument('--img_lambda_l1_ratio', type=float, default=0)

    group.add_argument('--lambda_outside_loss', type=float, default=1e-3)
    group.add_argument('--lambda_alpha_lap_loss', type=float, default=0)
    group.add_argument('--lambda_no_surf_init_density_lap_loss', type=float, default=0)
    group.add_argument('--lambda_normal_loss', type=float, default=0)
    group.add_argument('--surf_normal_loss_lambda_type', type=str, default='const', 
                        choices=['const', 'linear'])
    group.add_argument('--lambda_normal_loss_final', type=float, default=0)
    group.add_argument('--lambda_normal_loss_delay_steps', type=float, default=0)
    group.add_argument('--lambda_normal_loss_decay_steps', type=float, default=0)

    group.add_argument('--lambda_surf_sign_loss', type=float, default=0)
    group.add_argument('--lambda_surface_eikonal', type=float, default=0)
    group.add_argument('--alpha_weighted_norm_loss', action='store_true', default=False,
                        help='Use alpha value to re-weight the normal loss')
    group.add_argument('--py_surf_norm_reg', action='store_true', default=False,
                        help='Use Pytorch version of surface normal regularization')
    # Foreground TV
    group.add_argument('--lambda_tv', type=float, default=1e-5)
    group.add_argument('--lambda_tv_alpha', type=float, default=1e-5)
    group.add_argument('--lambda_tv_surface', type=float, default=0)
    group.add_argument('--tv_sparsity', type=float, default=0.01)
    group.add_argument('--alpha_lap_sparsity', type=float, default=0.01)
    group.add_argument('--tv_surface_sparsity', type=float, default=0.01)
    group.add_argument('--norm_surface_sparsity', type=float, default=0.01)
    group.add_argument('--no_surf_norm_con_check', action='store_true', default=False,
                    help='Do not check surface connectivity when computing surface norm regularization')
    group.add_argument('--surf_norm_reg_ignore_empty', action='store_true', default=False,
                    help='Do not apply surface normal reg if two voxels are both empty')
    group.add_argument('--surf_norm_reg_l1', action='store_true', default=False,
                    help='Do not apply surface normal reg if two voxels are both empty')
    group.add_argument('--fused_surf_norm_reg', action='store_true', default=False,
                    help='Used fused surface normal regularization')
    group.add_argument('--tv_logalpha', action='store_true', default=False,
                    help='Use log(1-exp(-delta * sigma)) as in neural volumes')

    group.add_argument('--lambda_l_dist', type=float, default=0.0)
    group.add_argument('--lambda_l_entropy', type=float, default=0.0)
    group.add_argument('--lambda_l_samp_dist', type=float, default=0.0)
    group.add_argument('--lambda_sparsify_alpha', type=float, default=
                        0.0,
                        help="Weight for sparsity loss on log alpha. Used for surface optimization. Note that it works differently to plenoxel sparsity")
    group.add_argument('--delay_sparsify_alpha', type=float, default=0.0)
    
    group.add_argument('--lambda_sparsify_surf', type=float, default=
                        0.0,
                        help="Weight for sparsity loss on log surface. ")
    group.add_argument('--delay_sparsify_surf', type=float, default=0.0)
    group.add_argument('--sparsify_surf_decrease', action='store_true', default=False,
                        help="Sparsifying surface by decreasing the values ")
    group.add_argument('--sparsify_surf_thresh', type=float, default=0.1, help='Alpha threshold for surface sparsity to be applied')
    group.add_argument('--alpha_surf_sparsify_sparsity', type=float, default=0.01)
    group.add_argument('--alpha_sparsify_bound', type=float, default=1e-6)
    group.add_argument('--surf_sparsify_bound', type=float, default=-10)
    group.add_argument('--sparsify_only_trained_cells', action='store_true', default=False)

    group.add_argument('--lambda_viscosity_loss', type=float, default=0)
    group.add_argument('--viscosity_sparsity', type=float, default=0.1)
    group.add_argument('--viscosity_eta', type=float, default=1e-2)
    

    group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
    group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

    group.add_argument('--lambda_tv_lumisphere', type=float, default=0.0)#1e-2)#1e-3)
    group.add_argument('--tv_lumisphere_sparsity', type=float, default=0.01)
    group.add_argument('--tv_lumisphere_dir_factor', type=float, default=0.0)

    group.add_argument('--tv_decay', type=float, default=1.0)

    group.add_argument('--lambda_l2_sh', type=float, default=0.0)#1e-4)
    group.add_argument('--tv_early_only', type=int, default=1, help="Turn off TV regularization after the first split/prune")

    group.add_argument('--tv_contiguous', type=int, default=1,
                            help="Apply TV only on contiguous link chunks, which is faster")
    # End Foreground TV

    group.add_argument('--lambda_sparsity', type=float, default=
                        0.0,
                        help="Weight for sparsity loss as in SNeRG/PlenOctrees " +
                            "(but applied on the ray)")
    group.add_argument('--lambda_beta', type=float, default=
                        0.0,
                        help="Weight for beta distribution sparsity loss as in neural volumes")


    # Background TV
    group.add_argument('--lambda_tv_background_sigma', type=float, default=1e-2)
    group.add_argument('--lambda_tv_background_color', type=float, default=1e-2)

    group.add_argument('--tv_background_sparsity', type=float, default=0.01)
    # End Background TV

    # Basis TV
    group.add_argument('--lambda_tv_basis', type=float, default=0.0,
                    help='Learned basis total variation loss')
    # End Basis TV

    group.add_argument('--weight_decay_sigma', type=float, default=1.0)
    group.add_argument('--weight_decay_sh', type=float, default=1.0)

    group.add_argument('--lr_decay', action='store_true', default=True)

    group.add_argument('--n_train', type=lambda x: int(float(x)), default=None, help='Number of training images. Defaults to use all avaiable.')

    group.add_argument('--n_eval_train', type=lambda x: int(float(x)), default=1, help='Number of train images to be evaluated and logged')
    group.add_argument('--n_eval_test', type=lambda x: int(float(x)), default=1, help='Number of test images to be evaluated and logged')

    group.add_argument('--nosphereinit', action='store_true', default=False,
                        help='do not start with sphere bounds (please do not use for 360)')

    group.add_argument('--load_ckpt', action='store_true', default=False,
                        help='resume training from ckpt in the given path if exists')

    group.add_argument('--surface_init', type=str, default=None)



    group.add_argument('--eval_cf', action='store_true', default=False)
    group.add_argument('--surf_eval_n_sample', type=int, default=10)
    group.add_argument('--surf_eval_intersect_th', type=float, default=0.1)


    # group.add_argument('--log_tune_hparam_config_path', type=str, default=None,
    #                    help='Log hyperparamters being tuned to tensorboard based on givn config.json path')

    if return_parpser:
        return parser

    args = parser.parse_args()
    # maybe_merge_config_file(args)

    return args
