from .nerf_dataset import NeRFDataset
from .llff_dataset import LLFFDataset
from .nsvf_dataset import NSVFDataset
from .co3d_dataset import CO3DDataset
from os import path

def auto_dataset(root : str, *args, **kwargs):
    if path.isfile(path.join(root, 'apple', 'eval_batches_multisequence.json')):
        print("Detected CO3D dataset")
        return CO3DDataset(root, *args, **kwargs)
    elif path.isfile(path.join(root, 'poses_bounds.npy')):
        print("Detected LLFF dataset")
        return LLFFDataset(root, *args, **kwargs)
    elif path.isfile(path.join(root, 'transforms.json')) or \
         path.isfile(path.join(root, 'transforms_train.json')):
        print("Detected NeRF (Blender) dataset")
        return NeRFDataset(root, *args, **kwargs)
    else:
        print("Defaulting to extended NSVF dataset")
        return NSVFDataset(root, *args, **kwargs)

datasets = {
    'nerf': NeRFDataset,
    'llff': LLFFDataset,
    'nsvf': NSVFDataset,
    'co3d': CO3DDataset,
    'auto': auto_dataset
}
