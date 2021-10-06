import svox2
import svox
import math
import argparse
from os import path
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)
args = parser.parse_args()

grid = svox2.SparseGrid.load(args.ckpt)
t = grid.to_svox1()
print(t)

out_path = path.splitext(args.ckpt)[0] + '_svox1.npz'
print('Saving', out_path)
t.save(out_path)
