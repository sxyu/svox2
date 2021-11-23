"""
Process COLMAP output into NSVF data format (almost) ready to use with our system.
Usage: <colmap_root_dir>/sparse/0 (or replace 0 with some other partial map)

Add -s <number> to tune the world scale (default 1). You can tune this by
view_colmap_data.py or other tools.

NOTE: This file should probably be merged into the run_colmap.py

IMPORTANT: The dataset will not be split into train/test sets as currently required by
our code.  You need to append 1_ before test set image names and 0_ before train set image names.
Use python create_split.py <colmap_root_dir> to split it automatically if lazy.

*******

The root directory will look like

COLMAP preconditions:
images/  : all the images (expect this to already exist)
sparse/0 : the sparse map (from COLMAP)
database.db

Our main outputs:
pose/           : 4x4 pose matrix for each image
intrinsics.txt  : 4x4 intrinsics matrix, only 0,0 and 1,1 entries (focal length) matter

Additionally,
points.npy      : Nx3 sparse point cloud of features
feature/        : npz for each frame storing info about features in each image.
                  Contains fields xys and ids, where xys = position of feature on image,
                  ids = row of this point in points.npy

*******

If you want to pre-downscale the images, save the corresponding image files
in directory
images_<factor>/
where <factor> is something like 2 or 4. Then set data.factor in the config.
You can use mogrify to do this. Or use the script downsample.py, which uses OpenCV and concurrent.futures:

python downsample.py <colmap_root_dir>/images <factor>

Note that if you do not do this and set a factor in the config anyway,
the images will be resized dynamically on load.

*******

The code to parse the COLMAP sparse bin files is from LLFF.
"""
# Copyright 2021 Alex Yu
import os
import os.path as osp
import numpy as np
import struct
import collections
import argparse
import shutil

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_colmap_sparse(sparse_path):
    cameras = []
    with open(osp.join(sparse_path, "cameras.bin"), "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        assert num_cameras == 1, "Only supports single camera"
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            assert model_name in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"], \
                   "Only SIMPLE_PINHOLE/SIMPLE_RADIAL supported"
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras.append(
                Camera(
                    id=camera_id,
                    model=model_name,
                    width=width,
                    height=height,
                    params=np.array(params),
                )
            )
        assert len(cameras) == num_cameras
    points3D_idmap = {}
    points3D = []
    with open(osp.join(sparse_path, "points3D.bin"), "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for i in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            points3D_idmap[point3D_id] = i
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D.append(
                Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
            )
    images = []
    with open(osp.join(sparse_path, "images.bin"), "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = list(map(int, x_y_id_s[2::3]))
            point3D_ids = [points3D_idmap[x] for x in point3D_ids if x >= 0]
            point3D_ids = np.array(point3D_ids)
            images.append(
                Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
            )
    return cameras, images, points3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sparse_dir",
        type=str,
        help="COLMAP output sparse model dir e.g. sparse/0. We expect images to be at sparse_dir/../../images",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="Scale to apply to scene, tune this to improve the autoscaling",
    )
    parser.add_argument(
        "--gl_cam",
        action="store_true",
        default=False,
        help="Change camera space convention to match NeRF, jaxNeRF "
        "(our implementation uses OpenCV convention and does not need this, "
        "set data.gl_cam_space = True in the config if you use this option)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite output dirs if exists",
    )
    parser.add_argument(
        "--overwrite_no_del",
        action="store_true",
        default=False,
        help="Do not delete existing files if overwritting",
    )
    parser.add_argument(
        "--colmap_suffix",
        action="store_true",
        default=False,
        help="Output to pose_colmap and intrinsics_colmap.txt to retain the gt poses/intrinsics if available",
    )
    args = parser.parse_args()

    if args.sparse_dir.endswith("/"):
        args.sparse_dir = args.sparse_dir[:-1]
    base_dir = osp.dirname(osp.dirname(args.sparse_dir))
    pose_dir = osp.join(base_dir, "pose_colmap" if args.colmap_suffix else "pose")
    feat_dir = osp.join(base_dir, "feature")
    base_scale_file = osp.join(base_dir, "base_scale.txt")
    if osp.exists(base_scale_file):
        with open(base_scale_file, 'r') as f:
            base_scale = float(f.read())
        print('base_scale', base_scale)
    else:
        base_scale = 1.0
        print('base_scale defaulted to', base_scale)
    print("BASE_DIR", base_dir)
    print("POSE_DIR", pose_dir)
    print("FEATURE_DIR", feat_dir)
    print("COLMAP_OUT_DIR", args.sparse_dir)
    overwrite = args.overwrite

    def create_or_recreate_dir(dirname):
        if osp.isdir(dirname):
            import click

            nonlocal overwrite
            if overwrite or click.confirm(f"Directory {dirname} exists, overwrite?"):
                if not args.overwrite_no_del:
                    shutil.rmtree(dirname)
                overwrite = True
            else:
                print("Quitting")
                import sys

                sys.exit(1)
        os.makedirs(dirname, exist_ok=True)

    cameras, imdata, points3D = read_colmap_sparse(args.sparse_dir)
    create_or_recreate_dir(pose_dir)
    create_or_recreate_dir(feat_dir)

    print("Get intrinsics")
    K = np.eye(4)
    K[0, 0] = cameras[0].params[0] / base_scale
    K[1, 1] = cameras[0].params[0] / base_scale
    K[0, 2] = cameras[0].params[1] / base_scale
    K[1, 2] = cameras[0].params[2] / base_scale
    print("f", K[0, 0], "c", K[0:2, 2])
    np.savetxt(osp.join(base_dir, "intrinsics_colmap.txt" if args.colmap_suffix else "intrinsics.txt"), K)
    del K

    print("Get world scaling")
    points = np.stack([p.xyz for p in points3D])
    cen = np.median(points, axis=0)
    points -= cen
    dists = (points ** 2).sum(axis=1)

    # FIXME: Questionable autoscaling. Adopt method from Noah Snavely
    meddist = np.median(dists)
    points *= 2 * args.scale / meddist

    # Save the sparse point cloud
    np.save(osp.join(base_dir, "points.npy"), points)
    print(cen, meddist)

    print("Get cameras")

    bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])
    coord_trans = np.diag([1, -1, -1, 1.0])
    for im in imdata:
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        xys = im.xys
        point3d_ids = im.point3D_ids
        #  w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        t_world = -R.T @ t
        t_world = (t_world - cen[:, None]) * 2 * args.scale / meddist
        c2w = np.concatenate([np.concatenate([R.T, t_world], 1), bottom], 0)

        if args.gl_cam:
            # Use the alternate camera space convention of jaxNeRF, OpenGL etc
            # We use OpenCV convention
            c2w = c2w @ coord_trans

        imfile_name = osp.splitext(osp.basename(im.name))[0]
        pose_path = osp.join(pose_dir, imfile_name + ".txt")
        feat_path = osp.join(feat_dir, imfile_name + ".npz")  # NOT USED but maybe nice?
        np.savetxt(pose_path, c2w)
        np.savez(feat_path, xys=xys, ids=point3d_ids)
    print(" Total cameras:", len(imdata))
    print("Done!")


if __name__ == "__main__":
    main()
