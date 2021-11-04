"""
Splits dataset using NSVF conventions.
Every eighth image is used as a test image (1_ prefix) and other images are train (0_ prefix)

Usage:
python create_split.py <data_set_root>
data_set_root should contain directories like images/, pose/
"""
# Copyright 2021 Alex Yu
import os
import os.path as osp
import click
from typing import NamedTuple, List
import argparse
import random

parser = argparse.ArgumentParser("Automatic dataset splitting")
parser.add_argument('root_dir', type=str, help="COLMAP dataset root dir")
parser.add_argument('--every', type=int, default=8, help="Every x images used for testing")
parser.add_argument('--dry_run', action='store_true', help="Dry run, prints renames without modifying any files")
parser.add_argument('--random', action='store_true', help="If set, chooses the split randomly rather than at a fixed interval "
                                                          "(but number of images in train/test set is same)")
args = parser.parse_args()

class Dir(NamedTuple):
    name: str
    valid_exts: List[str]

def list_filter_dirs(base):
    all_dirs = [x for x in os.listdir(base) if osp.isdir(osp.join(base, x))]
    image_exts = [".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp"]
    depth_exts = [".exr", ".pfm", ".png", ".npy"]
    dirs_prefixes = [Dir(name="pose", valid_exts=[".txt"]),
                     Dir(name="poses", valid_exts=[".txt"]),
                     Dir(name="feature", valid_exts=[".npz"]),
                     Dir(name="rgb", valid_exts=image_exts),
                     Dir(name="images", valid_exts=image_exts),
                     Dir(name="image", valid_exts=image_exts),
                     Dir(name="c2w", valid_exts=image_exts),
                     Dir(name="depths", valid_exts=depth_exts)]
    dirs = []
    dir_idx = 0
    for pfx in dirs_prefixes:
        for d in all_dirs:
            if d.startswith(pfx.name):
                if d == "pose":
                    dir_idx = len(dirs)
                dirs.append(Dir(name=osp.join(base, d), valid_exts=pfx.valid_exts))
    return dirs, dir_idx

dirs, dir_idx = list_filter_dirs(args.root_dir)

refdir = dirs[dir_idx]
print("going to split", [x.name for x in dirs], "reference", refdir.name)
if args.dry_run or click.confirm("Continue?", default=True):
    filedata = {}
    base_files = [osp.splitext(x)[0] for x in sorted(os.listdir(refdir.name))
                  if osp.splitext(x)[1].lower() in refdir.valid_exts]
    if args.random:
        print('random enabled')
        random.shuffle(base_files)
    base_files_map = {x: f"{int(i % args.every == 0)}_" + x for i, x in enumerate(base_files)}

    for dir_obj in dirs:
        dirname = dir_obj.name
        files = sorted(os.listdir(dirname))
        for filename in files:
            full_filename = osp.join(dirname, filename)
            if filename.startswith("0_") or filename.startswith("1_"):
                continue
            if not osp.isfile(full_filename):
                continue
            base_file, ext = osp.splitext(filename)
            if ext.lower() not in dir_obj.valid_exts:
                print('SKIP ', full_filename, ' Since it has an unsupported extension')
                continue
            if base_file not in base_files_map:
                print('SKIP ', full_filename, ' Since it does not match any reference file')
                continue
            new_base_file = base_files_map[base_file]
            new_full_filename = osp.join(dirname, new_base_file + ext)
            print('rename', full_filename, 'to', new_full_filename)
            if not args.dry_run:
                os.rename(full_filename, new_full_filename)
    if args.dry_run:
        print('(dry run complete)')
    else:
        print('use unsplit.py to undo this operation')
