"""
Basic data visualization (using PlenOctree's volrend)
Usage: python view_data.py <data_root>
default output: data_vis.html. You can open this in your browser. (bash sensei/mkweb)
"""
# Copyright 2021 Alex Yu
import sys
import os
from os import path

DIR_PATH = path.dirname(os.path.realpath(__file__))
sys.path.append(path.join(DIR_PATH, ".."))

import warnings
import numpy as np
import math
from argparse import ArgumentParser
from nerfvis import Scene  # pip install nerfvis
from scipy.spatial.transform import Rotation

# BEGIN BORROWED CODE
# Copyright (c) 2006, Christoph Gohlke
# Copyright (c) 2006-2009, The Regents of the University of California
# All rights reserved.
def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis. """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.  """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0,  0.0),
                     (0.0,  cosa, 0.0),
                     (0.0,  0.0,  cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(((0.0,         -direction[2],  direction[1]),
                      (direction[2], 0.0,          -direction[0]),
                      (-direction[1], direction[0],  0.0)),
                     dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def get_best_yaw(C):
    '''
    maximize trace(Rz(theta) * C)
    '''
    assert C.shape == (3, 3)

    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)

    return theta

def rot_z(theta):
    R = tfs.rotation_matrix(theta, [0, 0, 1])
    R = R[0:3, 0:3]

    return R

def align_umeyama(model, data, known_scale=False, yaw_only=False):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type

    Output:
    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)
    t_error -- translational error per point (1xn)

    """

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)

    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if(np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
        S[2, 2] = -1

    if yaw_only:
        rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
        theta = get_best_yaw(rot_C)
        R = rot_z(theta)
    else:
        R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0/sigma2*np.trace(np.dot(D_svd, S))

    t = mu_M-s*np.dot(R, mu_D)

    return s, R, t

def align_procrustes_rt(t_a : np.ndarray, q_a : np.ndarray,
                        t_ref : np.ndarray,
                        use_first_k : int = 1000000,
                        want_transform : bool = False):
    """
    Align translation +  rotation
    :param t_a: camera translations to align (N, 3)
    :param q_a: camera rotations to align (xyz axis-angle, xyzw quaternion, or rotation matrix) (N, {3, 4, 9})
    :param t_ref: reference camera translations (N, 3)
    :param use_first_k: int, if set, uses only first k number of cameras to align
    :param want_transform: bool, if set, returns transform function instead of transformed points
    :return:
        if want_transform == False:
            t (N, 3), q (N, {3, 4, 9}) similarity-transformed version of cameraa poses, aligned to ref
        else: function which given points, applies the aligning transform
    """
    assert t_ref.shape[0] == t_a.shape[0]
    s, R, t = align_umeyama(t_ref[:use_first_k], t_a[:use_first_k])

    #  # Advanced alignment
    #  n_points = t_a.shape[0]
    #  z = np.zeros((n_points, 3))
    #  z[:, -1] = 0.05
    #  t_a_aug = t_a + quaternion_rotate_vector_np(q_a, z) / s
    #  t_ref_aug = t_ref + quaternion_rotate_vector_np(q_ref, z)
    #
    #  _, R, t = align_umeyama(np.concatenate([t_ref, t_ref_aug], axis=0), np.concatenate([t_a * s, t_a_aug * s], axis=0), known_scale=True)

    def transform(t_b : np.ndarray, q_b : np.ndarray):
        t_align = s * t_b @ R.T + t
        Ra = Rotation.from_matrix(R)
        q_align = (Ra * Rotation.from_matrix(q_b)).as_matrix()
        return t_align, q_align
    return transform if want_transform else transform(t_a, q_a)

# END BORROWED CODE

def get_image_size(path : str):
    """
    Get image size without loading it
    """
    from PIL import Image
    im = Image.open(path)
    return im.size # W, H

def sort_key(x):
    if len(x) > 2 and x[1] == "_":
        return x[2:]
    return x

def main():
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str, help="dataset root")
    parser.add_argument(
        "--seg",
        action="store_true",
        default=False,
        help="connect camera trajectories with lines, should be used e.g. in NeRF synthetic",
    )
    parser.add_argument(
        "--n_cameras_for_procrustes", '-P',
        type=int,
        default=100000,
        help="use at most first x cameras for procrustes. Useful if trajectory starts to diverge",
    )
    args = parser.parse_args()

    dataset_name = path.basename(path.abspath(args.data_dir))

    def look_for_dir(cands, required=True):
        for cand in cands:
            if path.isdir(path.join(args.data_dir, cand)):
                return path.join(args.data_dir, cand)
        if required:
            assert False, "None of " + str(cands) + " found in data directory"
        return ""

    pose_dir = path.join(args.data_dir, "pose_colmap")
    pose_gt_dir = look_for_dir(["poses", "pose", "c2w", "cameras"])
    if not path.isdir(pose_dir):
        pose_dir, pose_gt_dir = pose_gt_dir, None
    images_dir = look_for_dir(["images", "image", "rgb", "color", "rgbs"])
    intrin_path = path.join(args.data_dir, "intrinsics.txt")
    point_cloud_path = path.join(args.data_dir, "points.npy")

    print("POSE_DIR", pose_dir)
    print("IMAGES_PATH", images_dir)
    print("INTRIN_PATH", intrin_path)
    print("POINT_CLOUD_PATH", point_cloud_path)
    pose_files = sorted([x for x in os.listdir(pose_dir) if x.lower().endswith('.txt')], key=sort_key)
    image_files = sorted([x for x in os.listdir(images_dir) if x.lower().endswith('.png') or x.lower().endswith('.jpg')], key=sort_key)

    all_poses = []
    for i, pose_file in enumerate(pose_files):
        pose = np.loadtxt(path.join(pose_dir, pose_file)).reshape(4, 4)
        #  splt = path.splitext(pose_file)[0].split('_')
        #  num = int(splt[1] if len(splt) > 1 else splt[0])
        all_poses.append(pose)
    all_poses = np.stack(all_poses)

    def get_transform(c2w):
        t = c2w[:, :3, 3]
        R = c2w[:, :3, :3]

        # (1) Rotate the world so that z+ is the up axis
        # we estimate the up axis by averaging the camera up axes
        ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
        world_up = np.mean(ups, axis=0)
        world_up /= np.linalg.norm(world_up)

        up_camspace = np.array([0.0, -1.0, 0.0])
        c = (up_camspace * world_up).sum()
        cross = np.cross(world_up, up_camspace)
        skew = np.array([[0.0, -cross[2], cross[1]],
                         [cross[2], 0.0, -cross[0]],
                         [-cross[1], cross[0], 0.0]])
        R_align = np.eye(3)
        if c > -1:
            R_align = R_align + skew + (skew @ skew) * 1 / (1+c)
        else:
            # In the unlikely case the original data has y+ up axis,
            # rotate 180-deg about x axis
            R_align = np.array([[-1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])

        R = (R_align @ R)
        fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
        t = (R_align @ t[..., None])[..., 0]

        # (2) Recenter the scene using camera center rays
        # find the closest point to the origin for each camera's center ray
        dvec = t + (fwds * -t).sum(-1)[:, None] * fwds

        # Median for more robustness
        translate = -np.median(dvec, axis=0)

        transform = np.eye(4)
        transform[:3, 3] = translate
        transform[:3, :3] = R_align

        # (3) Rescale the scene using camera distances
        scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
        scale *= 0.95
        return transform, scale

    T, scale = get_transform(all_poses)
    all_poses = T @ all_poses

    R = all_poses[:, :3, :3]
    t = all_poses[:, :3, 3] * scale

    intrins = np.loadtxt(intrin_path)
    focal = (intrins[0, 0] + intrins[1, 1]) * 0.5
    image_wh = get_image_size(path.join(images_dir, image_files[0]))

    scene = Scene("colmap dataset: " + dataset_name)


    # Try to pick a good frustum size
    avg_dist : float = np.mean(np.linalg.norm(t[1:] - t[:-1], axis=-1))
    cam_scale = avg_dist * 0.3

    # Infer world up direction from GT cams
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    # Camera forward vector
    forwards = np.sum(R * np.array([0, 0, 1.0]), axis=-1)
    vforward = np.mean(forwards, axis=0)
    vforward /= np.linalg.norm(vforward)

    # Set camera center of rotation (origin) for orbit
    origin = np.mean(t, axis=0)

    # Set camera position
    center = origin - vforward * np.linalg.norm(t - origin, axis=-1).mean() * 0.7 * 3
    print('  camera center', center, 'vforward', vforward, 'world_up', world_up)

    scene.add_camera_frustum(name=f"traj_{i:04d}", focal_length=focal,
                             image_width=image_wh[0],
                             image_height=image_wh[1],
                             z=0.1,
                             r=R,
                             t=t,
                             connect=args.seg,
                             color=[1.0, 0.0, 0.0])

    if pose_gt_dir is not None:
        print('Loading GT')
        pose_gt_files = sorted([x for x in os.listdir(pose_gt_dir) if x.endswith('.txt')], key=sort_key)
        all_gt_poses = []
        for pose_file in pose_gt_files:
            pose = np.loadtxt(path.join(pose_gt_dir, pose_file))
            all_gt_poses.append(pose)
        all_gt_poses = np.stack(all_gt_poses)
        R_gt = all_gt_poses[:, :3, :3]
        t_gt = all_gt_poses[:, :3, 3]
        pose_files_st = set(pose_files)
        pose_gt_inds = np.array([i for i, pose_gt_file in enumerate(pose_gt_files) if pose_gt_file in pose_files_st], dtype=np.int64)
        print(len(pose_gt_inds), 'of', len(pose_gt_files), 'registered')
        if len(pose_gt_inds) < len(pose_gt_files):
            warnings.warn("Not all frames registered")

        r = R.reshape(-1, 9)
        r_gt = R_gt.reshape(-1, 9)

        transform = align_procrustes_rt(
                t_gt[pose_gt_inds], r_gt[pose_gt_inds],
                t, r, use_first_k=args.n_cameras_for_procrustes, want_transform=True)

        t_gt, r_gt = transform(t_gt, r_gt)
        R_gt = r_gt.reshape(-1, 3, 3)
        scene.add_camera_frustum(name=f"traj_gt", focal_length=focal,
                                 image_width=image_wh[0],
                                 image_height=image_wh[1],
                                 z=0.1,
                                 r=R_gt,
                                 t=t_gt,
                                 connect=args.seg,
                                 color=[0.0, 0.0, 1.0])
        scene.add_sphere(name=f"start", translation=t_gt[0],
                         scale=avg_dist * 0.1,
                         color=[0.0, 1.0, 1.0])

    if path.isfile(point_cloud_path):
        point_cloud = np.load(point_cloud_path)
        point_cloud = (T[:3, :3] @ point_cloud[:, :, None])[:, :, 0] + T[:3, 3]
        point_cloud *= scale
        scene.add_points("point_cloud", point_cloud, color=[0.0, 0.0, 0.0], unlit=True)


    out_dir = path.join(args.data_dir, "visual")
    scene.add_axes(length=1.0, visible=False)
    scene.add_sphere("Unit Sphere", visible=False)
    scene.add_cube("Unit Cube", scale=2, visible=False)
    print('WRITING', out_dir)
    scene.display(out_dir, world_up=world_up, cam_origin=origin, cam_center=center, cam_forward=vforward)



if __name__ == "__main__":
    main()
