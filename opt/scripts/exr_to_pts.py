# https://blender.stackexchange.com/questions/204886/calculating-3d-world-co-ordinates-using-depth-map-and-camera-intrinsics

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import json
from pathlib import Path
import numpy as np
import sklearn.neighbors as skln

scene_name = "materials"

data_json = f'/home/tw554/plenoxels/data/nerf_synthetic/{scene_name}/depth_render/transforms.json'
j = json.load(open(data_json, "r"))
frames = j['frames']
exr_dir = Path(f'/home/tw554/plenoxels/opt/scripts/home/tw554/plenoxels/data/nerf_synthetic/{scene_name}/depth_render')
out_path = str(Path(data_json).parent.parent / 'shape.npy')
print('output to:')
print(out_path)

thresh = 0.001

all_pts = []

for i in range(len(frames)):
    frame = frames[i]
    exr_path = sorted(list(exr_dir.glob(f'r_{i}_*.exr')))[0]
    depth = cv2.imread(str(exr_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    depth = depth[...,0]

    c2w = np.array(frame["transform_matrix"])


    # Distance factor from the cameral focal angle
    factor = 2.0 * np.tan(j["camera_angle_x"]/2.0)
    
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    # Valid depths are defined by the camera clipping planes
    valid = (depth < 1e10)
    
    # Negate Z (the camera Z is at the opposite)
    z = -np.where(valid, depth, np.nan)
    # Mirror X
    # Center c and r relatively to the image size cols and rows
    ratio = max(rows,cols)
    x = -np.where(valid, factor * z * (c - (cols / 2)) / ratio, 0)
    y = np.where(valid, factor * z * (r - (rows / 2)) / ratio, 0)
    
    points = np.dstack((x, y, z))[valid]

    pts = [c2w @ np.concatenate([p, [1]])[:, None] for p in points]

    all_pts.append(pts)

all_pts = np.concatenate(all_pts, axis=0)[:, :3, 0]

# downsample density
nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
nn_engine.fit(all_pts)
rnn_idxs = nn_engine.radius_neighbors(all_pts, radius=thresh, return_distance=False)
mask = np.ones(all_pts.shape[0], dtype=np.bool_)
for curr, idxs in enumerate(rnn_idxs):
    if mask[curr]:
        mask[idxs] = 0
        mask[curr] = 1
data_down = all_pts[mask]

np.save(out_path, data_down)

