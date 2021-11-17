# Originally from LLFF
# https://github.com/Fyusion/LLFF
# With minor modifications from NeX
# https://github.com/nex-mpi/nex-code

import numpy as np
import os
import imageio

def get_image_size(path : str):
    """
    Get image size without loading it
    """
    from PIL import Image
    im = Image.open(path)
    return im.size[1], im.size[0]    # H, W

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, "images_{}".format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, "images_{}x{}".format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, "images")
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [
        f
        for f in imgs
        if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
    ]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = "images_{}".format(r)
            resizearg = "{}%".format(100.0 / r)
        else:
            name = "images_{}x{}".format(r[1], r[0])
            resizearg = "{}x{}".format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print("Minifying", r, basedir)

        os.makedirs(imgdir)
        check_output("cp {}/* {}".format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split(".")[-1]
        args = " ".join(
            ["mogrify", "-resize", resizearg, "-format", "png", "*.{}".format(ext)]
        )
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != "png":
            check_output("rm {}/*.{}".format(imgdir, ext), shell=True)
            print("Removed duplicates")
        print("Done")


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))
    shape = 5

    # poss llff arr [3, 5, images] [R | T | intrinsic]
    # intrinsic same for all images
    if os.path.isfile(os.path.join(basedir, "hwf_cxcy.npy")):
        shape = 4
        # h, w, fx, fy, cx, cy
        intrinsic_arr = np.load(os.path.join(basedir, "hwf_cxcy.npy"))

    poses = poses_arr[:, :-2].reshape([-1, 3, shape]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    if not os.path.isfile(os.path.join(basedir, "hwf_cxcy.npy")):
        intrinsic_arr = poses[:, 4, 0]
        poses = poses[:, :4, :]

    img0 = [
        os.path.join(basedir, "images", f)
        for f in sorted(os.listdir(os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]
    sh = get_image_size(img0)

    sfx = ""
    if factor is not None:
        sfx = "_{}".format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, "images" + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, "does not exist, returning")
        return

    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    if poses.shape[-1] != len(imgfiles):
        print(
            "Mismatch between imgs {} and poses {} !!!!".format(
                len(imgfiles), poses.shape[-1]
            )
        )
        return

    if not load_imgs:
        return poses, bds, intrinsic_arr

    def imread(f):
        if f.endswith("png"):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255.0 for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print("Loaded image data", imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs, intrinsic_arr


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    # poses [images, 3, 4] not [images, 3, 5]
    # hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center)], 1)

    return c2w


def render_path_axis(c2w, up, ax, rad, focal, N):
    render_poses = []
    center = c2w[:, 3]
    hwf = c2w[:, 4:5]
    v = c2w[:, ax] * rad
    for t in np.linspace(-1.0, 1.0, N + 1)[:-1]:
        c = center + t * v
        z = normalize(c - (center - focal * c2w[:, 2]))
        # render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def render_path_spiral(c2w, up, rads, focal, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    # hwf = c2w[:,4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        # render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def recenter_poses(poses):
    # poses [images, 3, 4]
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)

    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def load_llff_data(
    basedir,
    factor=None,
    recenter=True,
    bd_factor=0.75,
    spherify=False,
    #  path_zflat=False,
    split_train_val=8,
    render_style="",
):

    # poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    poses, bds, intrinsic = _load_data(
        basedir, factor=factor, load_imgs=False
    )  # factor=8 downsamples original imgs by 8x

    print("Loaded LLFF data", basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    # poses [R | T] [3, 4, images]
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    # poses [3, 4, images] --> [images, 3, 4]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)

    # imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    # images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)
    else:
        c2w = poses_avg(poses)
        print("recentered", c2w.shape)

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        close_depth, inf_depth = -1, -1
        # Find a reasonable "focus depth" for this dataset
        #  if os.path.exists(os.path.join(basedir, "planes_spiral.txt")):
        #      with open(os.path.join(basedir, "planes_spiral.txt"), "r") as fi:
        #          data = [float(x) for x in fi.readline().split(" ")]
        #          dmin, dmax = data[:2]
        #          close_depth = dmin * 0.9
        #          inf_depth = dmax * 5.0
        #  elif os.path.exists(os.path.join(basedir, "planes.txt")):
        #      with open(os.path.join(basedir, "planes.txt"), "r") as fi:
        #          data = [float(x) for x in fi.readline().split(" ")]
        #          if len(data) == 3:
        #              dmin, dmax, invz = data
        #          elif len(data) == 4:
        #              dmin, dmax, invz, _ = data
        #          close_depth = dmin * 0.9
        #          inf_depth = dmax * 5.0

        prev_close, prev_inf = close_depth, inf_depth
        if close_depth < 0 or inf_depth < 0 or render_style == "llff":
            close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0

        if render_style == "shiny":
            close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
            if close_depth < prev_close:
                close_depth = prev_close
            if inf_depth > prev_inf:
                inf_depth = prev_inf

        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        #  if path_zflat:
        #      #             zloc = np.percentile(tt, 10, 0)[2]
        #      zloc = -close_depth * 0.1
        #      c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
        #      rads[2] = 0.0
        #      N_rots = 1
        #      N_views /= 2

        render_poses = render_path_spiral(
            c2w_path, up, rads, focal, zrate=0.5, rots=N_rots, N=N_views
        )

    render_poses = np.array(render_poses).astype(np.float32)
    # reference_view_id should stay in train set only
    validation_ids = np.arange(poses.shape[0])
    validation_ids[::split_train_val] = -1
    validation_ids = validation_ids < 0
    train_ids = np.logical_not(validation_ids)
    train_poses = poses[train_ids]
    train_bds = bds[train_ids]
    c2w = poses_avg(train_poses)

    dists = np.sum(np.square(c2w[:3, 3] - train_poses[:, :3, 3]), -1)
    reference_view_id = np.argmin(dists)
    reference_depth = train_bds[reference_view_id]
    print(reference_depth)

    return (
        reference_depth,
        reference_view_id,
        render_poses,
        poses,
        intrinsic
    )
