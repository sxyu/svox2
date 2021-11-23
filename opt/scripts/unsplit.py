"""
Inverse of create_split.py
"""
# Copyright 2021 Alex Yu
import os
import os.path as osp
import click
from typing import NamedTuple, List
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('root_dir', type=str, help="COLMAP dataset root dir")
parser.add_argument('--dry_run', action='store_true', help="Dry run, prints renames without modifying any files")
parser.add_argument('--yes', '-y', action='store_true', help="Answer yes")
args = parser.parse_args()

class Dir(NamedTuple):
    name: str
    valid_exts: List[str]

def list_filter_dirs(base):
    all_dirs = [x for x in os.listdir(base) if osp.isdir(osp.join(base, x))]
    image_exts = [".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp"]
    depth_exts = [".exr", ".pfm", ".png", ".npy"]
    dirs_prefixes = [Dir(name="pose", valid_exts=[".txt"]),
                     Dir(name="feature", valid_exts=[".npz"]),
                     Dir(name="rgb", valid_exts=image_exts),
                     Dir(name="images", valid_exts=image_exts),
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
print("going to unsplit", [x.name for x in dirs], "reference", dirs[dir_idx].name)
do_proceed = args.dry_run or args.yes
if not do_proceed:
    import click
    do_proceed = click.confirm("Continue?", default=True)
if do_proceed:
    filedata = {}
    base_files = [osp.splitext(x)[0] for x in sorted(os.listdir(refdir.name))
                  if osp.splitext(x)[1] in refdir.valid_exts and
                  (x.startswith('0_') or x.startswith('1_'))]
    base_files_map = {x: '_'.join(x.split('_')[1:]) for x in base_files}

    for dir_obj in dirs:
        dirname = dir_obj.name
        files = sorted(os.listdir(dirname))
        for filename in files:
            full_filename = osp.join(dirname, filename)
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
        print('use create_split.py to split again')
