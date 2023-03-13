import pyvista as pv
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import shutil


parser = argparse.ArgumentParser()

parser.add_argument('--input_path', default='')
parser.add_argument('--out_dir', default=None)
parser.add_argument('--is_mesh', action='store_true', default=False)
parser.add_argument('--no_color', action='store_true', default=False)
parser.add_argument('--n_imgs', type=int, default=30)
parser.add_argument('--crop_vid', action='store_true', default=False)

args = parser.parse_args()

if Path(args.out_dir).exists():
    shutil.rmtree(args.out_dir)

Path(args.out_dir).mkdir(exist_ok=True, parents=True)


obj = pv.read(args.input_path)

c2w_path = '/rds/project/rds-qxpdOeYWi78/plenoxels/data/nerf_synthetic/test_c2ws.npy'

for scene in ['ficus', 'drums', 'materials']:
    if scene in args.input_path:
        c2w_path = '/rds/project/rds-qxpdOeYWi78/plenoxels/data/nerf_synthetic/test_c2ws_nerf.npy'

c2ws = np.load(c2w_path)

mask = (obj.points < 1.5).all(axis=-1) & (obj.points > -1.5).all(axis=-1)

# if args.mask_crop:
#     filter_mask = ((obj.points > np.array([[0.1, 0.1, -100]])).all(axis=-1)) & ((obj.points < np.array([[100, 100, 0.]])).all(axis=-1))
#     mask = mask & (~filter_mask)

if args.crop_vid:
    c2w = c2ws[0]
    y_bounds = np.linspace(0.25, 0.12, args.n_imgs)

    for i, y_bound in enumerate(y_bounds):
        obj = pv.read(args.input_path)
        mask = (obj.points < 1.5).all(axis=-1) & (obj.points > -1.5).all(axis=-1) & (obj.points[:, 1] < y_bound)

        obj['mask'] = mask
        obj = obj.threshold(scalars='mask', value=True)

        img_size = (800, 800)
        background = 'white'

        focal_len = 1111

        t = c2w[:3,3]

        d = c2w[:3,:3] @ np.array([0,0,1])
        d = d / np.linalg.norm(d)
        up = (0,0,1)
        focal_point = t + d * focal_len

        cpos = (t, focal_point, up)
        if not args.no_color:
            obj.plot(scalars='RGB', rgb=True, cpos=cpos, 
                    screenshot=f'{args.out_dir}/{i:05d}.png', off_screen=True, eye_dome_lighting=True,
                    point_size=1, show_axes=False, background=background, window_size=img_size, zoom=0.75,
                    notebook=False,
                    )
        else:
            obj.plot(color='white', cpos=cpos, 
                    screenshot=f'{args.out_dir}/{i:05d}.png', off_screen=True, eye_dome_lighting=True,
                    point_size=1, show_axes=False, background=background, window_size=img_size, zoom=0.75,
                    notebook=False,
                    )



obj['mask'] = mask
obj = obj.threshold(scalars='mask', value=True)

img_size = (800, 800)
p = pv.Plotter(off_screen=True, notebook=False, window_size=img_size)
p.add_mesh(obj)

background = 'white'
p.set_background(background)




render_num = args.n_imgs
if render_num < 0:
    render_num = len(c2ws)

for i in range(0, len(c2ws), len(c2ws) // render_num):
    c2w = c2ws[i]
    focal_len = 1111

    t = c2w[:3,3]

    d = c2w[:3,:3] @ np.array([0,0,1])
    d = d / np.linalg.norm(d)
    up = (0,0,1)
    focal_point = t + d * focal_len


    if args.is_mesh:
        p.camera.position = t
        p.camera.focal = focal_point
        p.camera.up = up
        p.show(screenshot=f'{args.out_dir}/{i:05d}.png', auto_close=False)
    else:
        cpos = (t, focal_point, up)
        if not args.no_color:
            obj.plot(scalars='RGB', rgb=True, cpos=cpos, 
                    screenshot=f'{args.out_dir}/{i:05d}.png', off_screen=True, eye_dome_lighting=True,
                    point_size=1, show_axes=False, background=background, window_size=img_size, zoom=0.75,
                    notebook=False,
                    )
        else:
            obj.plot(color='white', cpos=cpos, 
                    screenshot=f'{args.out_dir}/{i:05d}.png', off_screen=True, eye_dome_lighting=True,
                    point_size=1, show_axes=False, background=background, window_size=img_size, zoom=0.75,
                    notebook=False,
                    )


