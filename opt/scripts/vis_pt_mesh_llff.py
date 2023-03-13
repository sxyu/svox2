import pyvista as pv
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import shutil
import cv2

import torch
import torch.nn.functional as F
import os
from glob import glob
import imageio
import pdb


from scipy.spatial.transform import Rotation
import struct
import glob
import copy

import os
from collections import deque
from typing import Union, Optional
from dataclasses import dataclass
from typing import Optional, Union, List

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--is_mesh', action='store_true', default=False)

    args = parser.parse_args()

    print(args)

    # if Path(args.out_dir).exists():
    #     shutil.rmtree(args.out_dir)

    Path(args.out_dir).mkdir(exist_ok=True, parents=True)


    obj = pv.read(args.input_path)

    # mask = (obj.points < 1.5).all(axis=-1) & (obj.points > -1.5).all(axis=-1)

    # if args.mask_crop:
    #     filter_mask = ((obj.points > np.array([[0.1, 0.1, -100]])).all(axis=-1)) & ((obj.points < np.array([[100, 100, 0.]])).all(axis=-1))
    #     mask = mask & (~filter_mask)

    # obj['mask'] = mask
    # obj = obj.threshold(scalars='mask', value=True)


    dset = LLFFDataset(args.dataset)

    img_size = (dset.w, dset.h)
    p = pv.Plotter(off_screen=True, notebook=False, window_size=img_size)

    background = 'white'
    p.set_background(background)

    for i in range(dset.c2w.shape[0]):
        c2w = dset.c2w[i].numpy()

        t = c2w[:3,3]

        d = c2w[:3,:3] @ np.array([0,0,1])
        d = d / np.linalg.norm(d)
        
        up = c2w[:3,:3] @ np.array([0,-1,0])
        up = up / np.linalg.norm(up)
        

        # apply ndc
        ndc_coeffs = dset.ndc_coeffs
        near = 1
        near_t = (near - t[2]) / d[2]
        t = t + near_t * d
        dx, dy, dz = d
        ox, oy, oz = t
        # Projection
        o0 = ndc_coeffs[0] * (ox / oz)
        o1 = ndc_coeffs[1] * (oy / oz)
        o2 = 1 - 2 * near / oz
        d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
        d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
        d2 = 2 * near / oz
        t = np.array([o0,o1,o2])
        d = np.array([d0,d1,d2])


        focal_point = t + d


        if args.is_mesh:
            p.add_mesh(obj)
            p.camera.position = t
            p.camera.focal = focal_point
            p.camera.up = up
            p.show(screenshot=f'{args.out_dir}/{i:05d}.png', auto_close=False, zoom=1.225)
        else:
            cpos = (t, focal_point, up)
            obj.plot(color='white', cpos=cpos, 
                    screenshot=f'{args.out_dir}/{i:05d}.png', off_screen=True, eye_dome_lighting=True,
                    point_size=1, show_axes=False, background=background, window_size=img_size, zoom=0.35,
                    notebook=False,
                    )






# LLFF-format Forward-facing dataset loader
# Please use the LLFF code to run COLMAP & convert 
#
# Adapted from NeX data loading code (NOT using their hand picked bounds)
# Entry point: LLFFDataset
#
# Original:
# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Distribute under MIT License

#from torch.utils.data import Dataset


@dataclass
class Intrin:
    fx: Union[float, torch.Tensor]
    fy: Union[float, torch.Tensor]
    cx: Union[float, torch.Tensor]
    cy: Union[float, torch.Tensor]

    def scale(self, scaling: float):
        return Intrin(
                    self.fx * scaling,
                    self.fy * scaling,
                    self.cx * scaling,
                    self.cy * scaling
                )

    def get(self, field:str, image_id:int=0):
        val = self.__dict__[field]
        return val if isinstance(val, float) else val[image_id].item()

@dataclass
class Rays:
    origins: Union[torch.Tensor, List[torch.Tensor]]
    dirs: Union[torch.Tensor, List[torch.Tensor]]
    gt: Union[torch.Tensor, List[torch.Tensor]]

    def to(self, *args, **kwargs):
        origins = self.origins.to(*args, **kwargs)
        dirs = self.dirs.to(*args, **kwargs)
        gt = self.gt.to(*args, **kwargs)
        return Rays(origins, dirs, gt)

    def __getitem__(self, key):
        origins = self.origins[key]
        dirs = self.dirs[key]
        gt = self.gt[key]
        return Rays(origins, dirs, gt)

    def __len__(self):
        return self.origins.size(0)


def convert_to_ndc(origins, directions, ndc_coeffs, near: float = 1.0):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane, not sure if needed
    t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)

    # Projection
    o0 = ndc_coeffs[0] * (ox / oz)
    o1 = ndc_coeffs[1] * (oy / oz)
    o2 = 1 - 2 * near / oz

    d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    d2 = 2 * near / oz

    origins = torch.stack([o0, o1, o2], -1)
    directions = torch.stack([d0, d1, d2], -1)
    return origins, directions

class LLFFDataset():
    """
    LLFF dataset loader adapted from NeX code
    Some arguments are inherited from them and not super useful in our case
    """
    def __init__(
        self,
        root,
        # root : str,
        # split : str,
        # epoch_size : Optional[int] = None,
        # device: Union[str, torch.device] = "cpu",
        # permutation: bool = True,
        # factor: int = 1,
        # ref_img: str="",
        scale : Optional[float]=1.0/4.0,  # 4x downsample
        dmin : float=-1,
        dmax : int=-1,
        invz : int= 0,
        # transform=None,
        render_style="",
        hold_every=0, #8,
        offset=250,
        # **kwargs
    ):
        super().__init__()
        # if scale is None:
        #     scale = 1.0 / 4.0  # Default 1/4 size for LLFF data since it's huge
        # self.scale = scale
        self.scale = scale

        self.dataset = root
        self.device = 'cuda:0'
        split = 'train'
        self.split = 'train'
        self.sfm = SfMData(
            root,
            ref_img="",
            dmin=dmin,
            dmax=dmax,
            invz=invz,
            scale=scale,
            render_style=render_style,
            offset=offset,
            hold_every=hold_every,
        )

        assert len(self.sfm.cams) == 1, \
                "Currently assuming 1 camera for simplicity, " \
                "please feel free to extend"
        self.imgs = []
        is_train_split = split.endswith('train')
        for i, ind in enumerate(self.sfm.imgs):
            img = self.sfm.imgs[ind]
            self.imgs.append(img)
            # if hold_every>0:
            #     img_train_split = ind % hold_every > 0
            #     if is_train_split == img_train_split:
            #         self.imgs.append(img)
            # else:
            #     img_train_split = ind - train_idx < 0
            #     if is_train_split == img_train_split:
            #         self.imgs.append(img)
                
        self.is_train_split = is_train_split

        self._load_images()
        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        assert self.h_full == self.sfm.ref_cam["height"]
        assert self.w_full == self.sfm.ref_cam["width"]

        self.intrins_full = Intrin(self.sfm.ref_cam['fx'],
                                   self.sfm.ref_cam['fy'],
                                   self.sfm.ref_cam['px'],
                                   self.sfm.ref_cam['py'])

        self.ndc_coeffs = (2 * self.intrins_full.fx / self.w_full,
                           2 * self.intrins_full.fy / self.h_full)


        self.h, self.w = self.h_full, self.w_full
        self.intrins = self.intrins_full

        self.focal = (self.intrins.fx + self.intrins.fy) / 2.


        self.intrinsics_all = []
        for i in range(self.n_images):
            intrinsics = np.eye(4)
            intrinsics[0][0]=self.focal
            intrinsics[1][1]=self.focal
            intrinsics[0][2] = self.intrins.cx
            intrinsics[1][2] = self.intrins.cy
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            # self.pose_all.append(torch.from_numpy(pose).float())

        
        # self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]

    


    def _load_images(self):
        scale = self.scale

        all_gt = []
        all_c2w = []
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        global_w2rc = np.concatenate([self.sfm.ref_img['r'], self.sfm.ref_img['t']], axis=1)
        global_w2rc = np.concatenate([global_w2rc, bottom], axis=0).astype(np.float64)
        for idx in tqdm(range(len(self.imgs))):
            R = self.imgs[idx]["R"].astype(np.float64)
            t = self.imgs[idx]["center"].astype(np.float64)
            c2w = np.concatenate([R, t], axis=1)
            c2w = np.concatenate([c2w, bottom], axis=0)
            #  c2w = global_w2rc @ c2w
            all_c2w.append(torch.from_numpy(c2w.astype(np.float32)))

            if 'path' in self.imgs[idx]:
                img_path = self.imgs[idx]["path"]
                img_path = os.path.join(self.dataset, img_path)
                if not os.path.isfile(img_path):
                    path_noext = os.path.splitext(img_path)[0]
                    # Hack: also try png
                    if os.path.exists(path_noext + '.png'):
                        img_path = path_noext + '.png'
                img = imageio.imread(img_path)
                if scale != 1 and not self.sfm.use_integral_scaling:
                    h, w = img.shape[:2]
                    if self.sfm.dataset_type == "deepview":
                        newh = int(h * scale)  # always floor down height
                        neww = round(w * scale)
                    else:
                        newh = round(h * scale)
                        neww = round(w * scale)
                    img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
                all_gt.append(torch.from_numpy(img))
        self.gt = torch.stack(all_gt).float() / 255.0
        if self.gt.size(-1) == 4:
            # Apply alpha channel
            self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
        self.c2w = torch.stack(all_c2w)
        bds_scale = 1.0
        self.z_bounds = [self.sfm.dmin * bds_scale, self.sfm.dmax * bds_scale]
        if bds_scale != 1.0:
            self.c2w[:, :3, 3] *= bds_scale

        if not self.is_train_split:
            render_c2w = []
            for idx in tqdm(range(len(self.sfm.render_poses))):
                R = self.sfm.render_poses[idx]["R"].astype(np.float64)
                t = self.sfm.render_poses[idx]["center"].astype(np.float64)
                c2w = np.concatenate([R, t], axis=1)
                c2w = np.concatenate([c2w, bottom], axis=0)
                render_c2w.append(torch.from_numpy(c2w.astype(np.float32)))
            self.render_c2w = torch.stack(render_c2w)
            if bds_scale != 1.0:
                self.render_c2w[:, :3, 3] *= bds_scale

        fx = self.sfm.ref_cam['fx']
        fy = self.sfm.ref_cam['fy']
        width = self.sfm.ref_cam['width']
        height = self.sfm.ref_cam['height']

        print('z_bounds from LLFF:', self.z_bounds, '(not used)')

        # Padded bounds
        radx = 1 + 2 * self.sfm.offset / self.gt.size(2)
        rady = 1 + 2 * self.sfm.offset / self.gt.size(1)
        radz = 1.0
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = [radx, rady, radz]
        print('scene_radius', self.scene_radius)
        self.use_sphere_bound = False



class SfMData:
    def __init__(
        self,
        root,
        ref_img="",
        scale=1,
        dmin=0,
        dmax=0,
        invz=0,
        render_style="",
        offset=200,
        hold_every=8,
    ):
        self.scale = scale
        self.ref_cam = None
        self.ref_img = None
        self.render_poses = None
        self.dmin = dmin
        self.dmax = dmax
        self.invz = invz
        self.dataset = root
        self.dataset_type = "unknown"
        self.render_style = render_style
        self.hold_every = hold_every
        self.white_background = False  # change background to white if transparent.
        self.index_split = []  # use for split dataset in blender
        self.offset = offset
        # Detect dataset type
        can_hanle = (
            self.readDeepview(root)
            or self.readLLFF(root, ref_img)
            or self.readColmap(root)
        )
        if not can_hanle:
            raise Exception("Unknow dataset type")
        # Dataset processing
        self.cleanImgs()
        self.selectRef(ref_img)
        self.scaleAll(scale)
        self.selectDepth(dmin, dmax, offset)

    def cleanImgs(self):
        """
        Remvoe non exist image from self.imgs
        """
        todel = []
        for image in self.imgs:
            img_path = self.dataset + "/" + self.imgs[image]["path"]
            if "center" not in self.imgs[image] or not os.path.exists(img_path):
                todel.append(image)
        for it in todel:
            del self.imgs[it]

    def selectRef(self, ref_img):
        """
        Select Reference image
        """
        if ref_img == "" and self.ref_cam is not None and self.ref_img is not None:
            return
        for img_id, img in self.imgs.items():
            if ref_img in img["path"]:
                self.ref_img = img
                self.ref_cam = self.cams[img["camera_id"]]
                return
        raise Exception("reference view not found")

    def selectDepth(self, dmin, dmax, offset):
        """
        Select dmin/dmax from planes.txt / bound.txt / argparse
        """
        if self.dmin < 0 or self.dmax < 0:
            if os.path.exists(self.dataset + "/bounds.txt"):
                with open(self.dataset + "/bounds.txt", "r") as fi:
                    data = [
                        np.reshape(np.matrix([float(y) for y in x.split(" ")]), [3, 1])
                        for x in fi.readlines()[3:]
                    ]
                ls = []
                for d in data:
                    v = self.ref_img["r"] * d + self.ref_img["t"]
                    ls.append(v[2])
                self.dmin = np.min(ls)
                self.dmax = np.max(ls)
                self.invz = 0

            elif os.path.exists(self.dataset + "/planes.txt"):
                with open(self.dataset + "/planes.txt", "r") as fi:
                    data = [float(x) for x in fi.readline().split(" ")]
                    if len(data) == 3:
                        self.dmin, self.dmax, self.invz = data
                    elif len(data) == 2:
                        self.dmin, self.dmax = data
                    elif len(data) == 4:
                        self.dmin, self.dmax, self.invz, self.offset = data
                        self.offset = int(self.offset)
                        print(f"Read offset from planes.txt: {self.offset}")
                    else:
                        raise Exception("Malform planes.txt")
            else:
                print("no planes.txt or bounds.txt found")
        if dmin > 0:
            print("Overriding dmin %f-> %f" % (self.dmin, dmin))
            self.dmin = dmin
        if dmax > 0:
            print("Overriding dmax %f-> %f" % (self.dmax, dmax))
            self.dmax = dmax
        if offset != 200:
            print(f"Overriding offset {self.offset}-> {offset}")
            self.offset = offset
        print(
            "dmin = %f, dmax = %f, invz = %d, offset = %d"
            % (self.dmin, self.dmax, self.invz, self.offset)
        )

    def readLLFF(self, dataset, ref_img=""):
        """
        Read LLFF
        Parameters:
          dataset (str): path to datasets
          ref_img (str): ref_image file name
        Returns:
          bool: return True if successful load LLFF data
        """
        if not os.path.exists(os.path.join(dataset, "poses_bounds.npy")):
            return False
        image_dir = os.path.join(dataset, "images")
        if not os.path.exists(image_dir) and not os.path.isdir(image_dir):
            return False

        self.use_integral_scaling = False
        scaled_img_dir = ''
        scale = self.scale
        if scale != 1 and abs((1.0 / scale) - round(1.0 / scale)) < 1e-9:
            # Integral scaling
            scaled_img_dir = "images_" + str(round(1.0 / scale))
            if os.path.isdir(os.path.join(self.dataset, scaled_img_dir)):
                self.use_integral_scaling = True
                image_dir = os.path.join(self.dataset, scaled_img_dir)
                print('Using pre-scaled images from', image_dir)
            else:
                scaled_img_dir = "images"
        else:
            scaled_img_dir = "images"

        # load R,T
        (
            reference_depth,
            reference_view_id,
            render_poses,
            poses,
            intrinsic
        ) = load_llff_data(
            dataset, factor=None, split_train_val=self.hold_every,
            render_style=self.render_style
        )

        # NSVF-compatible sort key
        def nsvf_sort_key(x):
            if len(x) > 2 and x[1] == '_':
                return x[2:]
            else:
                return x
        def keep_images(x):
            exts = ['.png', '.jpg', '.jpeg', '.exr']
            return [y for y in x if not y.startswith('.') and any((y.lower().endswith(ext) for ext in exts))] 

        # get all image of this dataset
        images_path = [os.path.join(scaled_img_dir, f) for f in sorted(keep_images(os.listdir(image_dir)), key=nsvf_sort_key)]

        # LLFF dataset has only single camera in dataset
        if len(intrinsic) == 3:
            H, W, f = intrinsic
            cx = W / 2.0
            cy = H / 2.0
            fx = f
            fy = f
        elif len(intrinsic) == 5:
            H, W, f, cx, cy = intrinsic
            fx = f
            fy = f
            print('our collected data intrinsic is ',H, W, fx,fy, cx, cy)
        else:
            H, W, fx, fy, cx, cy = intrinsic
        print('llff dataset intrinsic is',len(intrinsic), H, W, fx,fy, cx, cy)
        self.cams = {0: buildCamera(W, H, fx, fy, cx, cy)}
        print('poses,images_path',scaled_img_dir,poses.shape,len(images_path))
        # create render_poses for video render
        self.render_poses = buildNerfPoses(render_poses)

        # create imgs pytorch dataset
        # we store train and validation together
        # but it will sperate later by pytorch dataloader
        self.imgs = buildNerfPoses(poses, images_path)

        # if not set ref_cam, use LLFF ref_cam
        if ref_img == "":
            # restore image id back from reference_view_id
            # by adding missing validation index
            # if self.hold_every>0:
            #     print('our reference view id is',reference_view_id)
            #     image_id = reference_view_id + 1  # index 0 alway in validation set
            #     image_id = image_id + (image_id // self.hold_every)  # every 8 will be validation set
            #     self.ref_cam = self.cams[0]

            #     self.ref_img = self.imgs[image_id]  # here is reference view from train set
            #     print('we choose imageid',image_id)
            # else:
            print('our reference view id is',reference_view_id)
            image_id = reference_view_id
            self.ref_cam = self.cams[0]
            self.ref_img = self.imgs[image_id]


        # if not set dmin/dmax, use LLFF dmin/dmax
        if (self.dmin < 0 or self.dmax < 0) and (
            not os.path.exists(dataset + "/planes.txt")
        ):
            self.dmin = reference_depth[0]
            self.dmax = reference_depth[1]
        self.dataset_type = "llff"
        return True

    def scaleAll(self, scale):
        self.ocams = copy.deepcopy(self.cams)  # original camera
        for cam_id in self.cams.keys():
            cam = self.cams[cam_id]
            ocam = self.ocams[cam_id]

            nw = round(ocam["width"] * scale)
            nh = round(ocam["height"] * scale)
            sw = nw / ocam["width"]
            sh = nh / ocam["height"]
            cam["fx"] = ocam["fx"] * sw
            cam["fy"] = ocam["fy"] * sh
            # TODO: What is the correct way?
            #  cam["px"] = (ocam["px"] + 0.5) * sw - 0.5
            #  cam["py"] = (ocam["py"] + 0.5) * sh - 0.5
            cam["px"] = ocam["px"] * sw
            cam["py"] = ocam["py"] * sh
            cam["width"] = nw
            cam["height"] = nh

    def readDeepview(self, dataset):
        if not os.path.exists(os.path.join(dataset, "models.json")):
            return False

        self.cams, self.imgs = readCameraDeepview(dataset)
        self.dataset_type = "deepview"
        return True

    def readColmap(self, dataset):
        sparse_folder = dataset + "/dense/sparse/"
        image_folder = dataset + "/dense/images/"
        if (not os.path.exists(image_folder)) or (not os.path.exists(sparse_folder)):
            return False

        self.imgs = readImagesBinary(os.path.join(sparse_folder, "images.bin"))
        self.cams = readCamerasBinary(sparse_folder + "/cameras.bin")
        self.dataset_type = "colmap"
        return True


def readCameraDeepview(dataset):
    cams = {}
    imgs = {}
    with open(os.path.join(dataset, "models.json"), "r") as fi:
        js = json.load(fi)
        for i, cam in enumerate(js):
            for j, cam_info in enumerate(cam):
                img_id = cam_info["relative_path"]
                cam_id = img_id.split("/")[0]

                rotation = (
                    Rotation.from_rotvec(np.float32(cam_info["orientation"]))
                    .as_matrix()
                    .astype(np.float32)
                )
                position = np.array([cam_info["position"]], dtype="f").reshape(3, 1)

                if i == 0:
                    cams[cam_id] = {
                        "width": int(cam_info["width"]),
                        "height": int(cam_info["height"]),
                        "fx": cam_info["focal_length"],
                        "fy": cam_info["focal_length"] * cam_info["pixel_aspect_ratio"],
                        "px": cam_info["principal_point"][0],
                        "py": cam_info["principal_point"][1],
                    }
                imgs[img_id] = {
                    "camera_id": cam_id,
                    "r": rotation,
                    "t": -np.matmul(rotation, position),
                    "R": rotation.transpose(),
                    "center": position,
                    "path": cam_info["relative_path"],
                }
    return cams, imgs


def readImagesBinary(path):
    images = {}
    f = open(path, "rb")
    num_reg_images = struct.unpack("Q", f.read(8))[0]
    for i in range(num_reg_images):
        image_id = struct.unpack("I", f.read(4))[0]
        qv = np.fromfile(f, np.double, 4)

        tv = np.fromfile(f, np.double, 3)
        camera_id = struct.unpack("I", f.read(4))[0]

        name = ""
        name_char = -1
        while name_char != b"\x00":
            name_char = f.read(1)
            if name_char != b"\x00":
                name += name_char.decode("ascii")

        num_points2D = struct.unpack("Q", f.read(8))[0]

        for i in range(num_points2D):
            f.read(8 * 2)  # for x and y
            f.read(8)  # for point3d Iid

        r = Rotation.from_quat([qv[1], qv[2], qv[3], qv[0]]).as_dcm().astype(np.float32)
        t = tv.astype(np.float32).reshape(3, 1)

        R = np.transpose(r)
        center = -R @ t
        # storage is scalar first, from_quat takes scalar last.
        images[image_id] = {
            "camera_id": camera_id,
            "r": r,
            "t": t,
            "R": R,
            "center": center,
            "path": "dense/images/" + name,
        }

    f.close()
    return images


def readCamerasBinary(path):
    cams = {}
    f = open(path, "rb")
    num_cameras = struct.unpack("Q", f.read(8))[0]

    # becomes pinhole camera model , 4 parameters
    for i in range(num_cameras):
        camera_id = struct.unpack("I", f.read(4))[0]
        model_id = struct.unpack("i", f.read(4))[0]

        width = struct.unpack("Q", f.read(8))[0]
        height = struct.unpack("Q", f.read(8))[0]

        fx = struct.unpack("d", f.read(8))[0]
        fy = struct.unpack("d", f.read(8))[0]
        px = struct.unpack("d", f.read(8))[0]
        py = struct.unpack("d", f.read(8))[0]

        cams[camera_id] = {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "px": px,
            "py": py,
        }
        # fx, fy, cx, cy
    f.close()
    return cams


def nerf_pose_to_ours(cam):
    R = cam[:3, :3]
    center = cam[:3, 3].reshape([3, 1])
    center[1:] *= -1
    R[1:, 0] *= -1
    R[0, 1:] *= -1

    r = np.transpose(R)
    t = -r @ center
    return R, center, r, t


def buildCamera(W, H, fx, fy, cx, cy):
    return {
        "width": int(W),
        "height": int(H),
        "fx": float(fx),
        "fy": float(fy),
        "px": float(cx),
        "py": float(cy),
    }


def buildNerfPoses(poses, images_path=None):
    output = {}
    for poses_id in range(poses.shape[0]):
        R, center, r, t = nerf_pose_to_ours(poses[poses_id].astype(np.float32))
        output[poses_id] = {"camera_id": 0, "r": r, "t": t, "R": R, "center": center}
        if images_path is not None:
            output[poses_id]["path"] = images_path[poses_id]

    return output



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
    if poses_arr.shape[1] == 19:
        poses = poses_arr[:, :-4].reshape([-1, 3, shape]).transpose([1,2,0])
    else:
        poses = poses_arr[:, :-2].reshape([-1, 3, shape]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    if not os.path.isfile(os.path.join(basedir, "hwf_cxcy.npy")):
        intrinsic_arr = poses[:, 4, 0]
        poses = poses[:, :4, :]
        if poses_arr.shape[1] == 19:
            cx_cy = np.array(poses_arr[0,15:17])
            intrinsic_arr = np.concatenate((intrinsic_arr,cx_cy),axis=-1)
        print('our current intrinsic_arr is',intrinsic_arr)

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
    train_idx = 30
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
    # validation_ids = np.arange(poses.shape[0])
    # validation_ids[::split_train_val] = -1
    # validation_ids = validation_ids < 0
    # train_ids = np.logical_not(validation_ids)

    train_ids = np.arange(poses.shape[0])
    validation_ids = np.arange(poses.shape[0])
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


if __name__ == "__main__":
    main()