from .obj_dataset import NeRFDataset
from .ff_dataset import LLFFDataset

datasets = {
    'nerf': NeRFDataset,
    'llff': LLFFDataset
}
