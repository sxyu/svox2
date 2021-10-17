import svox2
import torch
import numpy as np
from util import Timing
from matplotlib import pyplot as plt
device='cuda:0'

GRID_FILE = 'lego.npy'
grid = svox2.SparseGrid(reso=256, device='cpu', radius=1.3256)
data = torch.from_numpy(np.load(GRID_FILE)).view(-1, grid.data_dim)
grid.sh_data.data = data[..., 1:]
grid.density_data.data = data[..., :1]
#  grid.resample(128, use_z_order=True)
grid = grid.cuda()

c2w = torch.tensor([
                [ -0.9999999403953552, 0.0, 0.0, 0.0 ],
                [ 0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708 ],
                [ 0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462 ],
                [ 0.0, 0.0, 0.0, 1.0 ],
            ], device=device)

with torch.no_grad():
    width = height = 800
    fx = fy = 1111
    origins = c2w[None, :3, 3].expand(height * width, -1).contiguous()
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=c2w.device),
        torch.arange(width, dtype=torch.float64, device=c2w.device),
    )
    xx = (xx - width * 0.5) / float(fx)
    yy = (yy - height * 0.5) / float(fy)
    zz = torch.ones_like(xx)
    dirs = torch.stack((xx, -yy, -zz), dim=-1)
    dirs /= torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs.reshape(-1, 3)
    del xx, yy, zz
    dirs = torch.matmul(c2w[None, :3, :3].double(), dirs[..., None])[..., 0].float()
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

    rays = svox2.Rays(origins, dirs)

    for i in range(5):
        with Timing("ours"):
            im = grid.volume_render(rays, use_kernel=True)

    im = im.reshape(height, width, 3)
    im = im.detach().clamp_(0.0, 1.0).cpu()
    plt.imshow(im)
    plt.show()
