# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np
from os import path
import pdb
         
DEBUG = False
            
VIEWS = 200
RESOLUTION = 500
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'

argv = sys.argv
argv = argv[argv.index("--") + 1:]
scene_name = argv[0]
out_path = f'/rds/project/rds-qxpdOeYWi78/plenoxels/data/nerf_synthetic/{scene_name}/demo/'

fp = bpy.path.abspath(out_path)


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

if not os.path.exists(fp):
    os.makedirs(fp)

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True


# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
#bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)


# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')

# depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
# depth_file_output.label = 'Depth Output'
# if FORMAT == 'OPEN_EXR':
#   links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
# else:
#   # Remap as other types can not represent the full range of depth.
#   map = tree.nodes.new(type="CompositorNodeMapValue")
#   # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
#   map.offset = [-0.7]
#   map.size = [DEPTH_SCALE]
#   map.use_min = True
#   map.min = [0]
#   links.new(render_layers.outputs['Depth'], map.inputs[0])

#   links.new(map.outputs[0], depth_file_output.inputs[0])

# normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
# normal_file_output.label = 'Normal Output'
# links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background


if scene_name != 'case':    
    objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
    bpy.ops.object.delete({"selected_objects": objs})

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

cam = scene.objects['Camera']
cam.location = (3., 3.0, 0.)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

from math import radians


#print(b_empty.rotation_euler)

azas = np.linspace(0,360, 11)[:10]
eles = [0, 45, -45]

out_data['frames'] = []

i = 0
for ele in eles:
    for aza in azas:
        
        b_empty.rotation_euler[0] = radians(ele)
        b_empty.rotation_euler[2] = radians(aza)
   
        scene.render.filepath = fp + '/r_' + str(i)


        bpy.ops.render.render(write_still=True)  # render still

        frame_data = {
            'file_path': scene.render.filepath,
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
        out_data['frames'].append(frame_data)

        i += 1

if not DEBUG:
    with open(fp + '/../' + 'transforms_demo.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)