# CO3D dataset loader
# https://github.com/facebookresearch/co3d/
#
# Adapted from basenerf
# Copyright 2021 Alex Yu
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from tqdm import tqdm
from os import path
import json
import gzip

from scipy.spatial.transform import Rotation
from typing import NamedTuple, Optional, List, Union
from .util import Rays, Intrin, similarity_from_cameras
from .dataset_base import DatasetBase


class CO3DDataset(DatasetBase):
    """
    CO3D Dataset
    Preloads all images for an object.
    Will create a data index on first load, to make later loads faster.
    """

    def __init__(
        self,
        root,
        split,
        seq_id : Optional[int] = None,
        epoch_size : Optional[int] = None,
        permutation: bool = True,
        device: Union[str, torch.device] = "cpu",
        max_image_dim: int = 800,
        max_pose_dist: float = 5.0,
        cam_scale_factor: float = 0.95,
        hold_every=8,
        **kwargs,
    ):
        """
        :param root: str dataset root directory
        :param device: data prefetch device
        """
        super().__init__()
        os.makedirs('co3d_tmp', exist_ok=True)
        index_file = path.join('co3d_tmp', 'co3d_index.npz')
        self.split = split
        self.permutation = permutation
        self.data_dir = root
        self.epoch_size = epoch_size
        self.max_image_dim = max_image_dim
        self.max_pose_dist = max_pose_dist
        self.cam_scale_factor = cam_scale_factor

        self.cats = sorted([x for x in os.listdir(root) if path.isdir(
            path.join(root, x))])
        self.gt = []
        self.n_images = 0
        self.curr_offset = 0
        self.next_offset = 0
        self.hold_every = hold_every
        self.curr_seq_cat = self.curr_seq_name = ''
        self.device = device
        if path.exists(index_file):
            print(' Using cached CO3D index', index_file)
            z = np.load(index_file)
            self.seq_cats = z.f.seq_cats
            self.seq_names = z.f.seq_names
            self.seq_offsets = z.f.seq_offsets
            self.all_image_size = z.f.image_size  # NOTE: w, h
            self.image_path = z.f.image_path
            self.image_pose = z.f.pose
            self.fxy = z.f.fxy
            self.cxy = z.f.cxy
        else:
            print(' Constructing CO3D index (1st run only), this may take a while')
            cam_trans = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32))
            frame_data_by_seq = {}
            self.seq_cats = []
            self.seq_names = []
            self.seq_offsets = []
            self.image_path = []
            self.all_image_size = []
            self.image_pose = []
            self.fxy = []
            self.cxy = []
            for i, cat in enumerate(self.cats):
                print(cat, '- category', i + 1, 'of', len(self.cats))
                cat_dir = path.join(root, cat)
                if not path.isdir(cat_dir):
                    continue
                frame_data_path = path.join(cat_dir, 'frame_annotations.jgz')
                with gzip.open(frame_data_path, 'r') as f:
                    all_frames_data = json.load(f)
                for frame_data in tqdm(all_frames_data):
                    seq_name = cat + '//' + frame_data['sequence_name']
                    #  frame_number = frame_data['frame_number']
                    if seq_name not in frame_data_by_seq:
                        frame_data_by_seq[seq_name] = []
                    pose = np.zeros((4, 4))
                    image_size_hw = frame_data['image']['size']  # H, W
                    H, W = image_size_hw
                    half_wh = np.array([W * 0.5, H * 0.5], dtype=np.float32)
                    R = np.array(frame_data['viewpoint']['R'])
                    T = np.array(frame_data['viewpoint']['T'])
                    fxy = np.array(frame_data['viewpoint']['focal_length'])
                    cxy = np.array(frame_data['viewpoint']['principal_point'])
                    focal = fxy * half_wh
                    prp = -1.0 * (cxy - 1.0) * half_wh
                    pose[:3, :3] = R
                    pose[:3, 3:] = -R @ T[..., None]
                    pose[3, 3] = 1.0
                    pose = pose @ cam_trans
                    frame_data_obj = {
                        'frame_number':frame_data['frame_number'],
                        'image_path':frame_data['image']['path'],
                        'image_size':np.array([W, H]),  # NOTE: this is w, h
                        'pose':pose,
                        'fxy':focal, # NOTE: this is x, y
                        'cxy':prp,   # NOTE: this is x, y
                    }
                    frame_data_by_seq[seq_name].append(frame_data_obj)
            print(' Sorting by sequence')
            for k in frame_data_by_seq:
                fd = sorted(frame_data_by_seq[k],
                        key=lambda x: x['frame_number'])
                spl = k.split('//')
                self.seq_cats.append(spl[0])
                self.seq_names.append(spl[1])
                self.seq_offsets.append(len(self.image_path))
                self.image_path.extend([x['image_path'] for x in fd])
                self.all_image_size.extend([x['image_size'] for x in fd])
                self.image_pose.extend([x['pose'] for x in fd])
                self.fxy.extend([x['fxy'] for x in fd])
                self.cxy.extend([x['cxy'] for x in fd])
            self.all_image_size = np.stack(self.all_image_size)
            self.image_pose = np.stack(self.image_pose)
            self.fxy = np.stack(self.fxy)
            self.cxy = np.stack(self.cxy)
            self.seq_offsets.append(len(self.image_path))
            self.seq_offsets = np.array(self.seq_offsets)
            print(' Saving to index')
            np.savez(index_file,
                    seq_cats=self.seq_cats,
                    seq_names=self.seq_names,
                    seq_offsets=self.seq_offsets,
                    image_size=self.all_image_size,
                    image_path=self.image_path,
                    pose=self.image_pose,
                    fxy=self.fxy,
                    cxy=self.cxy)
        self.n_seq = len(self.seq_names)
        print(
            " Loaded CO3D dataset",
            root,
            "n_seq", self.n_seq
        )

        if seq_id is not None:
            self.load_sequence(seq_id)


    def load_sequence(self, sequence_id : int):
        """
        Load a different CO3D sequence
        sequence_id should be at least 0 and at most (n_seq - 1)
        see co3d_tmp/co3d.txt for sequence ID -> name mappings
        """
        print('  Loading single CO3D sequence:',
                self.seq_cats[sequence_id], self.seq_names[sequence_id])
        self.curr_seq_cat = self.seq_cats[sequence_id]
        self.curr_seq_name = self.seq_names[sequence_id]
        self.curr_offset = self.seq_offsets[sequence_id]
        self.next_offset = self.seq_offsets[sequence_id + 1]
        self.gt = []
        fxs, fys, cxs, cys = [], [], [], []
        image_sizes = []
        c2ws = []
        ref_c2ws = []
        for i in tqdm(range(self.curr_offset, self.next_offset)):
            is_train = i % self.hold_every != 0
            ref_c2ws.append(self.image_pose[i])
            if self.split.endswith('train') != is_train:
                continue
            im = cv2.imread(path.join(self.data_dir, self.image_path[i]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            im = im[..., :3]
            h, w, _ = im.shape
            max_hw = max(h, w)
            approx_scale = self.max_image_dim / max_hw

            if approx_scale < 1.0:
                h2 = int(approx_scale * h)
                w2 = int(approx_scale * w)
                im = cv2.resize(im, (w2, h2), interpolation=cv2.INTER_AREA)
            else:
                h2 = h
                w2 = w
            scale = np.array([w2 / w, h2 / h], dtype=np.float32)
            image_sizes.append(np.array([h2, w2]))
            cxy = self.cxy[i] * scale
            fxy = self.fxy[i] * scale
            fxs.append(fxy[0])
            fys.append(fxy[1])
            cxs.append(cxy[0])
            cys.append(cxy[1])
            #  grid = data_util.gen_grid(h2, w2, cxy.astype(np.float32), normalize_scale=False)
            #  grid /= fxy.astype(np.float32)
            self.gt.append(torch.from_numpy(im))
            c2ws.append(self.image_pose[i])
        c2w = np.stack(c2ws, axis=0)
        ref_c2ws = np.stack(ref_c2ws, axis=0)  # For rescaling scene
        self.image_size = np.stack(image_sizes)
        fxs = torch.tensor(fxs)
        fys = torch.tensor(fys)
        cxs = torch.tensor(cxs)
        cys = torch.tensor(cys)

        # Filter out crazy poses
        dists = np.linalg.norm(c2w[:, :3, 3] - np.median(c2w[:, :3, 3], axis=0), axis=-1)
        med = np.median(dists)
        good_mask = dists < med * self.max_pose_dist
        c2w = c2w[good_mask]
        self.image_size = self.image_size[good_mask]
        good_idx = np.where(good_mask)[0]
        self.gt = [self.gt[i] for i in good_idx]

        self.intrins_full = Intrin(fxs[good_mask], fys[good_mask],
                cxs[good_mask], cys[good_mask])

        # Normalize
        #  c2w[:, :3, 3] -= np.mean(c2w[:, :3, 3], axis=0)
        #  dists = np.linalg.norm(c2w[:, :3, 3], axis=-1)
        #  c2w[:, :3, 3] *= self.cam_scale_factor / np.median(dists)

        T, sscale = similarity_from_cameras(ref_c2ws)
        c2w = T @ c2w
        c2w[:, :3, 3] *= self.cam_scale_factor * sscale

        self.c2w = torch.from_numpy(c2w).float()
        self.cam_n_rays = self.image_size[:, 0] * self.image_size[:, 1]
        self.n_images = len(self.gt)
        self.image_size_full = self.image_size

        if self.split == "train":
            self.gen_rays(factor=1)
        else:
            # Rays are not needed for testing
            self.intrins : Intrin = self.intrins_full


    def gen_rays(self, factor=1):
        print(" Generating rays, scaling factor", factor)
        # Generate rays
        self.factor = factor
        self.image_size = self.image_size_full // factor
        true_factor = self.image_size_full[:, 0] / self.image_size[:, 0]
        self.intrins = self.intrins_full.scale(1.0 / true_factor)

        all_origins = []
        all_dirs = []
        all_gts = []
        for i in tqdm(range(self.n_images)):
            yy, xx = torch.meshgrid(
                torch.arange(self.image_size[i, 0], dtype=torch.float32) + 0.5,
                torch.arange(self.image_size[i, 1], dtype=torch.float32) + 0.5,
            )
            xx = (xx - self.intrins.get('cx', i)) / self.intrins.get('fx', i)
            yy = (yy - self.intrins.get('cy', i)) / self.intrins.get('fy', i)
            zz = torch.ones_like(xx)
            dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)
            dirs = dirs.reshape(-1, 3, 1)
            del xx, yy, zz
            dirs = (self.c2w[i, None, :3, :3] @ dirs)[..., 0]

            if factor != 1:
                gt = F.interpolate(
                    self.gt[i].permute([2, 0, 1])[None], size=(self.image_size[i, 0],
                        self.image_size[i, 1]),
                    mode="area"
                )[0].permute([1, 2, 0])
                gt = gt.reshape(-1, 3)
            else:
                gt = self.gt[i].reshape(-1, 3)
            origins = self.c2w[i, None, :3, 3].expand(self.image_size[i, 0] *
                    self.image_size[i, 1], -1).contiguous()
            all_origins.append(origins)
            all_dirs.append(dirs)
            all_gts.append(gt)
        origins = all_origins
        dirs = all_dirs
        gt = all_gts

        if self.split == "train":
            origins = torch.cat([o.view(-1, 3) for o in origins], dim=0)
            dirs = torch.cat([o.view(-1, 3) for o in dirs], dim=0)
            gt = torch.cat([o.reshape(-1, 3) for o in gt], dim=0)

        self.rays_init = Rays(origins=origins, dirs=dirs, gt=gt)
        self.rays = self.rays_init
