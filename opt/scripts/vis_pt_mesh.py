import pyvista as pv
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument('--input_path', default='')
parser.add_argument('--out_dir', default=None)
parser.add_argument('--is_mesh', action='store_true', default=False)
parser.add_argument('--no_color', action='store_true', default=False)
parser.add_argument('--llff', action='store_true', default=False)
parser.add_argument('--extra_ele', type=float, default=None)
parser.add_argument('--mask_crop', action='store_true', default=False)

args = parser.parse_args()


azas = np.linspace(0,360, 11)[:10]
eles = [0, 30]

obj = pv.read(args.input_path)

mask = (obj.points < 1.5).all(axis=-1) & (obj.points > -1.5).all(axis=-1)

if args.mask_crop:
    filter_mask = ((obj.points > np.array([[0.1, 0.1, -100]])).all(axis=-1)) & ((obj.points < np.array([[100, 100, 0.]])).all(axis=-1))
    mask = mask & (~filter_mask)

obj['mask'] = mask
obj = obj.threshold(scalars='mask', value=True)

img_size = (500, 500)
p = pv.Plotter(off_screen=True, notebook=False, window_size=img_size)
p.add_mesh(obj)


background = 'white'
p.set_background(background)


if args.llff:
    # p.camera.position = (5, 5, 0)
    p.camera.position = (5, 5, 5)
    p.camera.up = (0.00359995, -0.99988694,  0.01459981)

    azas = [-15, 0, 15]
    eles = np.linspace(-150,-60, 11)[:10].tolist()
else:
    p.camera.position = (3, 3, 0)
    p.camera.focal_point = (0., 0., 0.)


i = 0

Path(args.out_dir).mkdir(exist_ok=True, parents=True)

if args.extra_ele is not None:
    eles.append(args.extra_ele)

for ele in eles:
    for aza in tqdm(azas):
        p.camera.azimuth = aza
        p.camera.elevation = ele


        if args.is_mesh:
            # cpos = p.show(screenshot=f'{args.out_dir}/{i:05d}.png', auto_close=False, return_cpos=True)
            p.show(screenshot=f'{args.out_dir}/{i:05d}.png', auto_close=False)
        else:
            cpos = (p.camera.position, p.camera.focal_point, p.camera.up)
            if not args.no_color:
                obj.plot(scalars='RGB', rgb=True, cpos=cpos, 
                        screenshot=f'{args.out_dir}/{i:05d}.png', off_screen=True, eye_dome_lighting=True,
                        point_size=1, show_axes=False, background=background, window_size=img_size, 
                        notebook=False,
                        )
            else:
                obj.plot(color='white', cpos=cpos, 
                        screenshot=f'{args.out_dir}/{i:05d}.png', off_screen=True, eye_dome_lighting=True,
                        point_size=1, show_axes=False, background=background, window_size=img_size, 
                        notebook=False,
                        )

        i+=1
