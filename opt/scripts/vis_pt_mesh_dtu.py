import pyvista as pv
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import shutil
import cv2


parser = argparse.ArgumentParser()

parser.add_argument('--input_path', default='')
parser.add_argument('--out_dir', default=None)
parser.add_argument('--scan', type=int, default=None)
parser.add_argument('--is_mesh', action='store_true', default=False)
parser.add_argument('--no_color', action='store_true', default=False)

args = parser.parse_args()

if Path(args.out_dir).exists():
    shutil.rmtree(args.out_dir)

Path(args.out_dir).mkdir(exist_ok=True, parents=True)


obj = pv.read(args.input_path)

# mask = (obj.points < 1.5).all(axis=-1) & (obj.points > -1.5).all(axis=-1)

# if args.mask_crop:
#     filter_mask = ((obj.points > np.array([[0.1, 0.1, -100]])).all(axis=-1)) & ((obj.points < np.array([[100, 100, 0.]])).all(axis=-1))
#     mask = mask & (~filter_mask)

# obj['mask'] = mask
# obj = obj.threshold(scalars='mask', value=True)

img_size = (800,600)
p = pv.Plotter(off_screen=True, notebook=False, window_size=img_size)


background = 'white'
p.set_background(background)

data_path = Path(f'/rds/project/rds-qxpdOeYWi78/plenoxels/data/dtu/dtu_scan{args.scan}') #/
n_imgs = len(list((data_path / 'image').glob('*')))

camera_dict = np.load(str(data_path / 'cameras_sphere.npz'))



def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose


for i in range(n_imgs):
    world_mat = camera_dict['world_mat_%d' % i].astype(np.float32)
    P = world_mat[:3, :4]
    intrinsics, c2w = load_K_Rt_from_P(None, P)

    t = c2w[:3,3]

    d = c2w[:3,:3] @ np.array([0,0,1])
    d = d / np.linalg.norm(d)
    
    up = c2w[:3,:3] @ np.array([0,-1,0])
    up = up / np.linalg.norm(up)
    
    focal_point = t + d


    if args.is_mesh:
        p.add_mesh(obj)
        p.camera.position = t
        p.camera.focal = focal_point
        p.camera.up = up
        p.show(screenshot=f'{args.out_dir}/{i:05d}.png', auto_close=False, zoom=1.225)
    else:
        cpos = (t, focal_point, up)
        if not args.no_color:
            obj.plot(scalars='RGB', rgb=True, cpos=cpos, 
                    screenshot=f'{args.out_dir}/{i:05d}.png', off_screen=True, eye_dome_lighting=True,
                    point_size=1, show_axes=False, background=background, window_size=img_size, zoom=1.225,
                    notebook=False,
                    )
        else:
            obj.plot(color='white', cpos=cpos, 
                    screenshot=f'{args.out_dir}/{i:05d}.png', off_screen=True, eye_dome_lighting=True,
                    point_size=1, show_axes=False, background=background, window_size=img_size, zoom=1.225,
                    notebook=False,
                    )


