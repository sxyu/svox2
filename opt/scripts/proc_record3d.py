import json
import argparse
import os
from os import path
import glob
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('--every', type=int, default=15)
parser.add_argument('--factor', type=int, default=2, help='downsample')
args = parser.parse_args()

video_file = glob.glob(args.data_dir + '/*.mp4')[0]
print('Video file:', video_file)
json_meta = path.join(args.data_dir, 'metadata.json')
meta = json.load(open(json_meta, 'r'))

K_3 = np.array(meta['K']).reshape(3, 3)
K = np.eye(4)
K[:3, :3] = K_3.T / args.factor
output_intrin_file = path.join(args.data_dir, 'intrinsics.txt')
np.savetxt(output_intrin_file, K)

poses = np.array(meta['poses'])

t = poses[:, 4:]
q = poses[:, :4]
R = Rotation.from_quat(q).as_matrix()

# Recenter the poses
center = np.mean(t, axis=0)
print('Scene center', center)
t -= center

all_poses = np.zeros((q.shape[0], 4, 4))
all_poses[:, -1, -1] = 1

Rt = np.concatenate([R, t[:, :, None]], axis=2)
all_poses[:, :3] = Rt
all_poses = all_poses @ np.diag([1, -1, -1, 1])
video = cv2.VideoCapture(str(video_file))
print(Rt.shape)

fps = video.get(cv2.CAP_PROP_FPS)
img_wh = ori_w, ori_h = (
    int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2,
    int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)

print('image size', img_wh)
pose_dir = path.join(args.data_dir, 'pose')
os.makedirs(pose_dir, exist_ok=True)

image_dir = path.join(args.data_dir, 'rgb')
os.makedirs(image_dir, exist_ok=True)
video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print('length', video_length)

idx = 0
for i in tqdm(range(0, video_length, args.every)):
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = video.read()
    if not ret or frame is None:
        print('skip', i)
        continue
    assert frame.shape[1] == img_wh[0] * 2
    assert frame.shape[0] == img_wh[1]
    frame = frame[:, img_wh[0]:]
    image_path = path.join(image_dir, f"{idx:05d}.png")
    pose_path = path.join(pose_dir, f"{idx:05d}.txt")

    if args.factor != 1:
        frame = cv2.resize(frame, (img_wh[0] // args.factor, img_wh[1] // args.factor), cv2.INTER_AREA)

    cv2.imwrite(image_path, frame)
    np.savetxt(pose_path, all_poses[i])
    idx += 1
