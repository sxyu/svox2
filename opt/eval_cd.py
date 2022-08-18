# evaluate chamfer distance

import numpy as np
import open3d as o3d
import copy
import torch
from chamferdist import ChamferDistance

pred_pts_path = '/home/tw554/plenoxels/opt/ckpt/lego/pts.npy'
gt_mesh_path = '/home/tw554/plenoxels/data/nerf_synthetic/lego/lego.obj'

pred_pts = np.load(pred_pts_path)
# down scale pred_pts
voxel_size = 0.01
pred_pts = pred_pts.voxel_down_sample(voxel_size)
radius_normal = voxel_size * 2
pred_pts.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
radius_feature = voxel_size * 5
pred_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pred_pts,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))


gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
# sample points
gt_pts = gt_mesh.sample_points_uniformly(10000)
# extract features
gt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    gt_pts,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

distance_threshold = voxel_size * 0.5
result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    pred_pts, gt_pts, pred_fpfh, gt_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=distance_threshold))

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])





