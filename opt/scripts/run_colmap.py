"""
Run COLMAP on a folder of images
Requires colmap installed
"""
# Copyright 2021 Oliver Wang (Adobe Research), with modifications by Alex Yu
# Similar version also found https://github.com/kwea123/nsff_pl/blob/master/preprocess.py 

import cv2
import moviepy
import moviepy.editor
import numpy
import argparse
import os
import random
import shutil
import sys
import tempfile
import torch
import torchvision
import glob
import numpy as np
from tqdm import tqdm
from warnings import warn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_colmap(strPath):
    # https://github.com/colmap/colmap/blob/master/src/ui/model_viewer_widget.cc#L71
    objIntrinsics = read_write_model.read_cameras_binary(strPath + '/cameras.bin')[1]

    objCameras = {}

    for intImage, objImage in enumerate(
            read_write_model.read_images_binary(strPath + '/images.bin').values()):

        npyIntrinsics = numpy.array(
            [[objIntrinsics.params[0], 0.0, objIntrinsics.params[1]],
             [0.0, objIntrinsics.params[0], objIntrinsics.params[2]], [0.0, 0.0, 1.0]],
            numpy.float32)
        npyExtrinsics = numpy.zeros([3, 4], numpy.float32)
        npyExtrinsics[0:3, 0:3] = read_write_model.qvec2rotmat(
            objImage.qvec / (numpy.linalg.norm(objImage.qvec) + 0.0000001))
        npyExtrinsics[0:3, 3] = objImage.tvec

        if objIntrinsics.model=='SIMPLE_RADIAL':
            objCameras[objImage.name] = {
                'model': objIntrinsics.model,
                'intIdent': objImage.id,
                'strImage': objImage.name,
                'dblFocal': objIntrinsics.params[0],
                'dblPrincipalX': objIntrinsics.params[1],
                'dblPrincipalY': objIntrinsics.params[2],
                'dblRadial': objIntrinsics.params[3],
                'npyIntrinsics': npyIntrinsics,
                'npyExtrinsics': npyExtrinsics,
                'intPoints': [intPoint for intPoint in objImage.point3D_ids if intPoint != -1]
            }
        elif objIntrinsics.model=='SIMPLE_PINHOLE':
            objCameras[objImage.name] = {
                'model': objIntrinsics.model,
                'intIdent': objImage.id,
                'strImage': objImage.name,
                'dblFocal': objIntrinsics.params[0],
                'dblPrincipalX': objIntrinsics.params[1],
                'dblPrincipalY': objIntrinsics.params[2],
                'npyIntrinsics': npyIntrinsics,
                'npyExtrinsics': npyExtrinsics,
                'intPoints': [intPoint for intPoint in objImage.point3D_ids if intPoint != -1]
            }

    objPoints = []

    for intPoint, objPoint in enumerate(
            read_write_model.read_points3D_binary(strPath + '/points3D.bin').values()):
        objPoints.append({
            'intIdent': objPoint.id,
            'npyLocation': objPoint.xyz,
            'npyColor': objPoint.rgb[::-1]
        })

    intPointindices = {}

    for intPoint, objPoint in enumerate(objPoints):
        intPointindices[objPoint['intIdent']] = intPoint

    for strCamera in objCameras:
        objCameras[strCamera]['intPoints'] = [
            intPointindices[intPoint] for intPoint in objCameras[strCamera]['intPoints']
        ]

    return objCameras, objPoints


def generate_masks(vid_root, args, overwrite=False):
    print('compute masks')
    vid_name = os.path.basename(vid_root)

    masks_dir = os.path.join(vid_root, args.mask_output)
    os.makedirs(masks_dir, exist_ok=True)

    frames_dir = os.path.join(vid_root, args.image_input)
    os.makedirs(frames_dir, exist_ok=True)

    maskrnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True).to(device).eval()

    files = sorted(
        glob.glob(os.path.join(vid_root, args.image_input, '*.jpg')) +
        glob.glob(os.path.join(vid_root, args.image_input, '*.png')))

    for file_ind, file in enumerate(tqdm(files, desc=f'masks: {vid_name}')):
        fn_ext = os.path.basename(file)
        fn = os.path.splitext(fn_ext)[0]
        frame_fn = f'{frames_dir}/{fn_ext}'
        out_mask_fn = f'{masks_dir}/{fn_ext}.png'

        if os.path.exists(out_mask_fn):
            continue

        im = cv2.imread(frame_fn)

        humans_tens = torch.FloatTensor(im.shape[0], im.shape[1]).fill_(1.0).to(device)

        obj_predictions = maskrnn_model(
            [torch.FloatTensor(im.transpose(2, 0, 1) / 255.0)[[2, 0, 1], :, :].to(device)])[0]

        for mask_ind in range(obj_predictions['masks'].size(0)):
            if obj_predictions['scores'][mask_ind].item() > 0.5:
                if obj_predictions['labels'][mask_ind].item() == 1:
                    humans_tens[obj_predictions['masks'][mask_ind, 0, :, :] > 0.5] = 0.0

                elif obj_predictions['labels'][mask_ind].item() == 31:
                    humans_tens[obj_predictions['masks'][mask_ind, 0, :, :] > 0.5] = 0.0

                elif obj_predictions['labels'][mask_ind].item() == 32:
                    humans_tens[obj_predictions['masks'][mask_ind, 0, :, :] > 0.5] = 0.0

                elif obj_predictions['labels'][mask_ind].item() == 48:
                    humans_tens[obj_predictions['masks'][mask_ind, 0, :, :] > 0.5] = 0.0

                # dog
                elif obj_predictions['labels'][mask_ind].item() == 18:
                    humans_tens[obj_predictions['masks'][mask_ind, 0, :, :] > 0.5] = 0.0

        mask_np = cv2.erode(
            src=humans_tens.cpu().numpy(),
            kernel=numpy.ones([3, 3], numpy.float32),
            anchor=(-1, -1),
            iterations=16,
            borderType=cv2.BORDER_DEFAULT)
        mask_np = (mask_np * 255.0).clip(0.0, 255.0).astype(numpy.uint8)

        cv2.imwrite(filename=out_mask_fn, img=mask_np)


def resize_frames(vid_root, args):
    vid_name = os.path.basename(vid_root)
    frames_dir = os.path.join(vid_root, args.images_resized)
    os.makedirs(frames_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(vid_root, args.image_input, '*.jpg')) +
        glob.glob(os.path.join(vid_root, args.image_input, '*.png')))

    print('Resizing images ...')
    factor = 1.0
    for file_ind, file in enumerate(tqdm(files, desc=f'imresize: {vid_name}')):
        out_frame_fn = f'{frames_dir}/{file_ind:05}.png'

        # skip if both the output frame and the mask exist
        if os.path.exists(out_frame_fn) and not overwrite:
            continue

        im = cv2.imread(file)

        # resize if too big
        if im.shape[1] > args.max_width or im.shape[0] > args.max_height:
            factor = max(im.shape[1] / args.max_width, im.shape[0] / args.max_height)
            dsize = (int(im.shape[1] / factor), int(im.shape[0] / factor))
            im = cv2.resize(src=im, dsize=dsize, interpolation=cv2.INTER_AREA)

        cv2.imwrite(out_frame_fn, im)
    return factor

def run_colmap(vid_root, args, factor, overwrite=False):
    max_num_matches = 132768
    overlap_frames = 75  # only used with sequential matching

    os.makedirs(os.path.join(vid_root, 'sparse'), exist_ok=True)

    extractor_cmd = f'''
        colmap feature_extractor \
            --database_path={vid_root}/database.db \
            --image_path={vid_root}/{args.images_resized}\
            --ImageReader.single_camera=1 \
            --ImageReader.default_focal_length_factor=0.69388 \
            --SiftExtraction.peak_threshold=0.004 \
            --SiftExtraction.max_num_features=8192 \
            --SiftExtraction.edge_threshold=16'''
    if args.noradial:
        extractor_cmd += ' --ImageReader.camera_model=SIMPLE_PINHOLE'
    else:
        extractor_cmd += ' --ImageReader.camera_model=SIMPLE_RADIAL'
    if args.use_masks:
        extractor_cmd += ' --ImageReader.mask_path={vid_root}/masks'
    known_intrin = False
    if args.known_intrin:
        intrin_path = os.path.join(vid_root, 'intrinsics.txt')
        if os.path.isfile(intrin_path):
            known_intrin = True
            print('Using known intrinsics')
            intrins = np.loadtxt(intrin_path)
            focal = (intrins[0, 0] + intrins[1, 1]) * 0.5 / factor
            cx, cy = intrins[0, 2] / factor, intrins[1, 2] / factor
            # f cx cy
            if args.noradial:
                extractor_cmd += f' --ImageReader.camera_params "{focal:.10f},{cx:.10f},{cy:.10f}"'
            else:
                extractor_cmd += f' --ImageReader.camera_params "{focal:.10f},{cx:.10f},{cy:.10f},0.0"'
        else:
            print('--known-intrin given but intrinsics.txt does not exist in data')
    os.system(extractor_cmd)

    if not args.do_sequential:
        os.system(f'''
            colmap exhaustive_matcher \
                --database_path={vid_root}/database.db \
                --SiftMatching.multiple_models=0 \
                --SiftMatching.max_ratio=0.8 \
                --SiftMatching.max_error=4.0 \
                --SiftMatching.max_distance=0.7 \
                --SiftMatching.max_num_matches={max_num_matches}''')
    else:
        warn("Using sequential matcher, which may be worse")
        os.system(f'''
            colmap sequential_matcher \
                --database_path={vid_root}/database.db \
                --SiftMatching.multiple_models=0 \
                --SiftMatching.max_num_matches={max_num_matches} \
                --SequentialMatching.overlap={overlap_frames} \
                --SequentialMatching.quadratic_overlap=0 \
                --SequentialMatching.loop_detection=1 \
                --SequentialMatching.vocab_tree_path={args.colmap_root}/vocab_tree_flickr100K_words256K.bin'''
                  )

    mapper_cmd = f'''
        colmap mapper \
            --database_path={vid_root}/database.db \
            --image_path={vid_root}/{args.images_resized} \
            --output_path={vid_root}/sparse '''

    if known_intrin and args.fix_intrin:
        mapper_cmd += f''' \
            --Mapper.ba_refine_focal_length=0 \
            --Mapper.ba_refine_principal_point=0 \
            --Mapper.ba_refine_extra_params=0 '''

    os.system(mapper_cmd)
    
    if not args.noradial:
        undist_dir = os.path.join(vid_root, args.undistorted_output)
        if not os.path.exists(undist_dir) or overwrite:
            os.makedirs(undist_dir, exist_ok=True)
            os.system(f'''
                colmap image_undistorter \
                    --input_path={vid_root}/sparse/0 \
                    --image_path={vid_root}/{args.images_resized} \
                    --output_path={vid_root} \
                    --output_type=COLMAP''')


def render_movie(vid_root, args):

    vid_name = os.path.basename(os.path.abspath(vid_root))
    files = sorted(glob.glob(os.path.join(vid_root, args.image_input , '*.png')) + glob.glob(os.path.join(vid_root, args.image_input , '*.jpg')))
    movie_fn = os.path.join(vid_root, f'{vid_name}_debug.mp4')

    #  if os.path.exists(movie_fn):
    #      print(f'{movie_fn} exists, skipping')
    #      return

    if not os.path.exists(os.path.join(vid_root, 'sparse', '0')):
        print(f'{vid_name} colmap model does not exist')
        return

    debug_dir = os.path.join(vid_root, 'debug', 'frames')
    os.makedirs(debug_dir, exist_ok=True)

    obj_cameras, obj_points = read_colmap(
        os.path.join(vid_root, 'sparse', '0'))

    for file_idx, file in enumerate(tqdm(files, desc=f'render: {vid_name}')):
        fn = os.path.basename(file)
        im = cv2.imread(file)

        if fn in obj_cameras:
            obj_camera = obj_cameras[fn]
            if obj_camera['model']=='SIMPLE_RADIAL':
                im = cv2.undistort(
                    src=im,
                    cameraMatrix=obj_camera['npyIntrinsics'],
                    distCoeffs=(obj_camera['dblRadial'], obj_camera['dblRadial'], 0.0, 0.0))
            elif obj_camera['model']=='SIMPLE_PINHOLE':
                im = cv2.undistort(
                    src=im,
                    cameraMatrix=obj_camera['npyIntrinsics'],
                    distCoeffs=(0.0,0.0,0.0,0.0))


            for obj_point in [obj_points[int_point] for int_point in obj_camera['intPoints']]:
                npyPoint = numpy.append(obj_point['npyLocation'], 1.0)
                npyPoint = numpy.matmul(obj_camera['npyIntrinsics'],
                                        numpy.matmul(obj_camera['npyExtrinsics'], npyPoint))
                if npyPoint[2] < 0.0000001: continue
                intX, intY = int(round(npyPoint[0] / npyPoint[2])), int(
                    round(npyPoint[1] / npyPoint[2]))
                if intX not in range(im.shape[1]) or intY not in range(im.shape[0]):
                    continue
                cv2.circle(img=im, center=(intX, intY), radius=1, color=(255, 0, 255), thickness=2)

        output_fn = f'{debug_dir}/{file_idx:05}.png'

        cv2.imwrite(filename=output_fn, img=im)

    # write movie
    ffmpeg_params = [
        '-crf', '5', '-pix_fmt', 'yuv420p', '-vf', 'pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2'
    ]
    moviepy.editor.ImageSequenceClip(
        sequence=debug_dir, fps=25).write_videofile(
            movie_fn, ffmpeg_params=ffmpeg_params)



def compute_poses(vid_root, args, overwrite=False):
    vid_name = os.path.basename(vid_root)
    colmap_dir = os.path.join(vid_root, 'sparse')
    pose_fn = os.path.join(vid_root, 'poses_bounds.npy')
    if not os.path.exists(pose_fn) or overwrite:
        print(f'poses: {vid_name}')
        # poses, pts3d, perm = load_colmap_data2(colmap_dir)
        # if poses is not None:
        #     save_poses(colmap_dir, poses, pts3d, perm)

        poses, pts3d, perm, save_arr = load_colmap_data(colmap_dir)
        if save_arr is not None:
            np.save(pose_fn, save_arr)


def preprocess(vid_root, args):
    print(f'processing: {vid_root}')

    frames_dir = os.path.join(vid_root, args.image_input)
    if not os.path.exists(frames_dir):
        files = os.listdir(vid_root)
        os.makedirs(frames_dir)
        print(f'Moving images to {frames_dir}')
        for fname in files:
            src_path = os.path.join(vid_root, fname)
            if not os.path.isfile(src_path):
                continue
            ext = os.path.splitext(fname)[1].upper()
            if ext == '.PNG' or ext == '.JPG' or ext == '.JPEG' or ext == '.EXR':
                os.rename(src_path, os.path.join(frames_dir, fname))

    overwrite = True
    factor = resize_frames(vid_root, args)
    # colmap
    if args.use_masks:
        generate_masks(vid_root, args, overwrite=overwrite)
    run_colmap(vid_root, args, factor, overwrite=overwrite)
    if args.debug:
        render_movie(vid_root, args)


if __name__ == '__main__':
    # method expects a folder of videos, each one has an image sequence in "frames"

    parser = argparse.ArgumentParser(description='Run COLMAP baseline')
    parser.add_argument(
        'vids', type=str, nargs='+', help='path to root with frames folder')
    parser.add_argument('--colmap-root', type=str, default='/home/sxyu/builds/colmap',
                help="COLMAP installation dir (only needed for vocab tree in case of sequential matcher)")
    parser.add_argument('--image-input', default='raw', help='location for source images')
    parser.add_argument('--mask-output', default='masks', help='location to store motion masks')
    parser.add_argument('--known-intrin', action='store_true', default=False, help='use intrinsics in <root>/intrinsics.txt if available')
    parser.add_argument('--fix-intrin', action='store_true', default=False, help='fix intrinsics in bundle adjustment, only used if --known-intrin is given and intrinsics.txt exists')
    parser.add_argument('--debug', action='store_true', default=False, help='render debug video')
    parser.add_argument('--noradial', action='store_true', default=False, help='do not use radial distortion')
    parser.add_argument('--use-masks', action='store_true', default=False, help='use automatic masks')
    parser.add_argument(
                    '--images-resized', default='images_resized', help='location for resized/renamed images')
    parser.add_argument(
        '--do-sequential', action='store_true', default=False, help='sequential rather than exhaustive matching')
    parser.add_argument('--max-width', type=int, default=1280, help='max image width')
    parser.add_argument('--max-height', type=int, default=768, help='max image height')
    parser.add_argument(
            '--undistorted-output', default='images', help='location of undistorted images')

    args = parser.parse_args()
    if args.noradial:
        args.images_resized = args.undistorted_output

    from vendor import read_write_model

    for vid in args.vids:
        preprocess(vid_root=vid, args=args)
