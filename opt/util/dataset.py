from .obj_dataset import NeRFDataset
from .ff_dataset import LLFFDataset
from .nsvf_dataset import NSVFDataset
from os import path

def auto_dataset(root : str, *args, **kwargs):
    if path.isfile(path.join(root, 'poses_bounds.npy')):
        print("Detected LLFF dataset")
        return LLFFDataset(root, *args, **kwargs)
    elif path.isfile(path.join(root, 'transforms.json')) or \
         path.isfile(path.join(root, 'transforms_train.json')):
        print("Detected NeRF dataset")
        return NeRFDataset(root, *args, **kwargs)
    else:
        print("Defaulting to NSVF dataset")
        return NSVFDataset(root, *args, **kwargs)

datasets = {
    'nerf': NeRFDataset,
    'llff': LLFFDataset,
    'nsvf': NSVFDataset,
    'auto': auto_dataset
}
