import pyvista as pv
from pyvista import examples
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument('--input_path', default='')
parser.add_argument('--out_dir', default=None)
parser.add_argument('--is_mesh', action='store_true', default=False)

args = parser.parse_args()


azas = np.linspace(0,360, 11)[:10]
eles = [0, 30]

obj = pv.read(args.input_path)

p = pv.Plotter(off_screen=True, notebook=False)
p.add_mesh(obj)
p.camera.position = (3, 3, 0)
p.camera.focal_point = (0., 0., 0.)

i = 0

Path(args.out_dir).mkdir(exist_ok=True, parents=True)

for ele in eles:
    for aza in tqdm(azas):
        p.camera.azimuth = aza
        p.camera.elevation = ele

        
        cpos = p.show(screenshot=f'{args.out_dir}/{i:05d}.png', auto_close=False, return_cpos=True)
        if not args.is_mesh:
            obj.plot(scalars='RGB', rgb=True, cpos=cpos, 
                    screenshot=f'{args.out_dir}/{i:05d}.png', off_screen=True, eye_dome_lighting=True,
                    point_size=1, show_axes=False, notebook=False
                    )

        i+=1
