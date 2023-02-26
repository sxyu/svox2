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

args = parser.parse_args()


azas = np.linspace(0,360, 11)[:10]
eles = [0, 30]

obj = pv.read(args.input_path)

mask = (obj.points < 1.5).all(axis=-1) & (obj.points > -1.5).all(axis=-1)
obj['mask'] = mask
obj = obj.threshold(scalars='mask', value=True)

img_size = (500, 500)
p = pv.Plotter(off_screen=True, notebook=False, window_size=img_size)
p.add_mesh(obj)
p.camera.position = (3, 3, 0)
p.camera.focal_point = (0., 0., 0.)

background = 'white'
p.set_background(background)

i = 0

Path(args.out_dir).mkdir(exist_ok=True, parents=True)

for ele in eles:
    for aza in tqdm(azas):
        p.camera.azimuth = aza
        p.camera.elevation = ele

        
        cpos = p.show(screenshot=f'{args.out_dir}/{i:05d}.png', auto_close=False, return_cpos=True)
        if not args.is_mesh:
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
