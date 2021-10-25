from .obj_dataset import NeRFDataset
from .ff_dataset import LLFFDataset
from os import path

def auto_dataset(root : str, *args, **kwargs):
    if path.isfile(path.join(root, 'poses_bounds.npy')):
        print("Detected LLFF dataset")
        return LLFFDataset(root, *args, **kwargs)
    else:
        print("Detected NeRF dataset")
        return NeRFDataset(root, *args, **kwargs)

datasets = {
    'nerf': NeRFDataset,
    'llff': LLFFDataset,
    'auto': auto_dataset
}
