import torch
import torch.cuda
from typing import NamedTuple, Optional, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor
    gt: torch.Tensor

    def to(self, *args, **kwargs):
        self.origins = self.origins.to(*args, **kwargs)
        self.dirs = self.dirs.to(*args, **kwargs)
        self.gt = self.gt.to(*args, **kwargs)

    def __getitem__(self, key):
        self.origins = self.origins[key]
        self.dirs = self.dirs[key]
        self.gt = self.gt[key]

class Timing:
    """
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, 'elapsed', self.start.elapsed_time(self.end), 'ms')


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps = 1000000):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """
    def helper(step):
        if lr_init == 0.0 and lr_final == 0.0:
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper
