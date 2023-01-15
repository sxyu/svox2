"""
Convert NeRF-iNGP data to NSVF
python ingp2nsvf.py <ngp_data_dir> <our_data_dir>
"""
import os
import shutil
from glob import glob
import json

import numpy as np
from PIL import Image
import argparse

def convert(data_dir : str, out_data_dir : str):
    """
    Convert Instant-NGP (modified NeRF) data to NSVF

    :param data_dir: the dataset dir (NeRF-NGP format) to convert
    :param out_data_dir: output dataset directory NSVF
    """

    images_dir_name = os.path.join(out_data_dir, "images")
    pose_dir_name = os.path.join(out_data_dir, "pose")

    os.makedirs(images_dir_name, exist_ok=True)
    os.makedirs(pose_dir_name, exist_ok=True)

    def get_subdir(name):
        if name.endswith("_train.json"):
            return "train"
        elif name.endswith("_val.json"):
            return "val"
        elif name.endswith("_test.json"):
            return "test"
        return ""

    def get_out_prefix(name):
        if name.endswith("_train.json"):
            return "0_"
        elif name.endswith("_val.json"):
            return "1_"
        elif name.endswith("_test.json"):
            return "2_"
        return ""

    jsons = {
        x: (get_subdir(x), get_out_prefix(x))
        for x in glob(os.path.join(data_dir, "*.json"))
    }

    # OpenGL -> OpenCV
    cam_trans = np.diag(np.array([1.0, -1.0, -1.0, 1.0]))

    # fmt: off
    world_trans = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # fmt: on

    assert len(jsons) > 0, f"No jsons found in {data_dir}, can't convert"
    cnt = 0

    example_fpath = None
    tj = {}
    for tj_path, (tj_subdir, tj_out_prefix) in jsons.items():
        with open(tj_path, "r") as f:
            tj = json.load(f)
        if "frames" not in tj:
            print(f"No frames in json {tj_path}, skipping")
            continue

        for frame in tj["frames"]:
            # Try direct relative path (used in newer NGP datasets)
            fpath = os.path.join(data_dir, frame["file_path"])
            if not os.path.isfile(fpath):
                # Legacy path (NeRF)
                fpath = os.path.join(
                    data_dir, tj_subdir, os.path.basename(frame["file_path"]) + ".png"
                )
            example_fpath = fpath
            if not os.path.isfile(fpath):
                print("Could not find image:", frame["file_path"], "(this may be ok)")
                continue

            ext = os.path.splitext(fpath)[1]

            c2w = np.array(frame["transform_matrix"])
            c2w = world_trans @ c2w @ cam_trans  # To OpenCV

            image_fname = tj_out_prefix + f"{cnt:05d}"

            pose_path = os.path.join(pose_dir_name, image_fname + ".txt")

            # Save 4x4 OpenCV C2W pose
            np.savetxt(pose_path, c2w)

            # Copy images
            new_fpath = os.path.join(images_dir_name, image_fname + ext)
            shutil.copyfile(fpath, new_fpath)
            cnt += 1

    assert len(tj) > 0, f"No valid jsons found in {data_dir}, can't convert"

    w = tj.get("w")
    h = tj.get("h")

    if w is None or h is None:
        assert example_fpath is not None
        # Pose not available so load a image and get the size
        w, h = Image.open(example_fpath).size

    fx = float(0.5 * w / np.tan(0.5 * tj["camera_angle_x"]))
    if "camera_angle_y" in tj:
        fy = float(0.5 * h / np.tan(0.5 * tj["camera_angle_y"]))
    else:
        fy = fx

    cx = tj.get("cx", w * 0.5)
    cy = tj.get("cy", h * 0.5)

    intrin_mtx = np.array([
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    # Write intrinsics
    np.savetxt(os.path.join(out_data_dir, "intrinsics.txt"), intrin_mtx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="NeRF-NGP data directory")
    parser.add_argument("out_data_dir", type=str, help="Output NSVF data directory")
    args = parser.parse_args()
    convert(args.data_dir, args.out_data_dir)
