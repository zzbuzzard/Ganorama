import torch
from PIL import Image
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
from whatis import whatis as wi
from tqdm import tqdm
import torchvision.transforms.functional as trf
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F
import open3d as o3d
import random
from typing import List
import gc

from pers2equir import MultiPerspective, MultiPerspectiveWeighted

# 1) For each image, project coordinates into 3D (requires casting a ray from given rotation)
#     B x H x W x 3d (or 4d)
# 2) For i in batch, for j in batch
# 3) Project i's 3d coords into j's 2d space
# 4) Collect pixel values from j's 2d space

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_intrinsic(fov: float, H: int, W: int) -> torch.Tensor:
    """
    Returns a 3×4 intrinsic matrix K = [ [f,0,cx,0],
                                         [0,f,cy,0],
                                         [0,0, 1,0] ]
    where f = (W/2)/tan(FOV/2).
    """
    f = (W / 2) / math.tan(math.radians(fov) / 2)
    cx, cy = (W-1) / 2, (H-1) / 2
    K = torch.tensor([
        [f, 0, cx, 0],
        [0, f, cy, 0],
        [0, 0,  1, 0],
    ], dtype=torch.float32, device=device)
    return K


def extrinsic(azimuth: float, elevation: float) -> torch.Tensor:
    """
    Returns a 4×4 camera‐to‐world pose (i.e. extrinsic) matrix:
      [ R | t ]
      [ 0 | 1 ]
    here t=0 (camera at origin), and R = R_z(azimuth) @ R_x(elevation).
    """
    ca, sa = math.cos(azimuth), math.sin(azimuth)
    ce, se = math.cos(elevation), math.sin(elevation)
    # Rz = torch.tensor([[ ca, -sa, 0],
    #                    [ sa,  ca, 0],
    #                    [  0,   0, 1]], device=device)
    Ry = torch.tensor([[ca, 0, sa],
                       [0,  1, 0],
                       [-sa,   0, ca]], device=device)
    Rx = torch.tensor([[1,   0,    0],
                       [0,  ce,  -se],
                       [0,  se,   ce]], device=device)
    R = Ry @ Rx  # rotate elevation -> rotate azimuth
    E = torch.eye(4, device=device)
    E[:3, :3] = R
    # no translation (camera at world‐origin)
    return E

fov = 90
device = 'cuda' if torch.cuda.is_available() else 'cpu'

azimuths = [i * 90 for i in range(4)] * 2 + [i * 45 for i in range(8)]
elevations = [45] * 4 + [-45] * 4 + [0] * 8
# azimuths = [i * 45 for i in range(B)]
# elevations = [45 for i in range(B)]

B = len(azimuths)

poses = [extrinsic(math.radians(a), math.radians(e)) for a, e in zip(azimuths, elevations)]  # B x 4 x 4
# inverse_poses = [extrinsic(math.radians(-a), math.radians(-e)) for a, e in zip(azimuths, elevations)]  # B x 4 x 4
inverse_poses = [torch.linalg.inv(M) for M in poses]  # B x 4 x 4


def vis(positions3d: List[torch.Tensor], color_list: List[torch.Tensor] = None):
    n = len(positions3d)
    random.seed(0)
    cols = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]] + [[random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)] for _ in range(100)]
    pcd_list=[]

    pcd_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    for i in range(n):
        pos = positions3d[i]

        # Homogeneous coords
        if pos.shape[-1] == 4:
            pos = pos.reshape(-1, 4)
            pos = pos[:, :3] / pos[:, 3:]
        else:
            pos = pos.reshape(-1, 3)
        pos = pos.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos)

        if color_list is None:
            pcd.paint_uniform_color(cols[i])
        else:
            c = color_list[i].cpu().numpy().reshape(-1, 3)
            pcd.colors = o3d.utility.Vector3dVector(c)

        pcd_list.append(pcd)

    o3d.visualization.draw_geometries(pcd_list)


# Returns tensor of shape (B x H x W x 4)
def get_3d_points(B, H, W, device=device):
    positions3d = []

    # 1) build pixel grid u,v
    u = torch.linspace(0, W-1, W, device=device).float()
    v = torch.linspace(0, H-1, H, device=device).float()
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # uu is x coord, vv is y coord

    # intrinsic params
    f = (W / 2) / math.tan(math.radians(fov) / 2)
    cx, cy = (W-1) / 2, (H-1) / 2
    # f = K[0, 0]
    # cx, cy = K[0, 2], K[1, 2]

    # back‐project to camera‐space rays @ z=1
    x = (uu - cx) / f
    y = (vv - cy) / f
    ones = torch.ones_like(x)
    cam_rays = torch.stack([x, y, ones, ones], dim=-1)  # H x W x 4
    # Compute 3D positions of each feature
    for i in range(B):
        # Apply inverse pose to ray to get 3D position
        v = poses[i] @ cam_rays.reshape(-1, 4).t()  # 4 x N
        # Adjust depth to be on surface of a sphere
        v[:3] /= torch.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        # Reshape (4 x N) -> (H x W x 4)
        pos = v.t().reshape(H, W, 4)

        # K = get_intrinsic(fov, H, W)  # 3 x 4
        # test = K @ poses[i] @ v

        positions3d.append(pos)

    return positions3d


@torch.no_grad()
def align(m, i, features: torch.Tensor) -> torch.Tensor:
    """
    Given:
      Ps       : list of B projection matrices (3×4)
      features : B×C×H×W  (per‐view feature maps)
    Returns:
      pos      : B×H×W×4   homogeneous world‐coords of each pixel (unit depth)
    """
    features = features.to(torch.float32)

    B, C, H, W = features.shape
    device = features.device

    K = get_intrinsic(fov, H, W)  # 3 x 4
    Ps = [K @ pose for pose in inverse_poses]            # list of B (3 x 4) tensors

    # print(f"Running for B={B}, C={C}, size = {H} x {W}")

    positions3d = get_3d_points(B, H, W, device)
    # vis(positions3d[:3])

    means = [i.mean() for i in features]
    stds = [i.std() for i in features]
    new_out = features.clone()
    counts = torch.ones((B, 1, H, W), device=device)
    agg_means = torch.zeros((B, 1, H, W), device=device)
    agg_std = torch.zeros((B, 1, H, W), device=device)
    agg_std2 = torch.zeros((B, 1, H, W), device=device)
    for i in range(B):
        agg_means[i] += means[i]
        agg_std[i] += stds[i]
        agg_std2[i] += stds[i] ** 2
        for j in range(B):
            if i == j:
                continue
            # Project positions3d[i] into view j, to find pixels in view j which correspond to our pixels
            # H x W x 4 -> 4 x N -> 3 x N
            uvw = Ps[j] @ positions3d[i].reshape(-1, 4).t()
            uv = uvw[:2] / uvw[2:]  # and normalise to pixel coords (2 x N)

            depth = uvw.t().reshape(H, W, 3)[..., 2]

            # H x W x 2
            coords = uv.t().reshape(H, W, 2)
            valid_inds = (coords[..., 0] >= 0) & (coords[..., 0] <= H-1) & (coords[..., 1] >= 0) & (coords[..., 1] <= W-1)  # H x W
            valid_inds = valid_inds & (depth > 0)

            # Prepare normalised_coords for F.grid_sample, which expects a slightly different format
            # Normalise from [0, H-1] to [-1, 1]
            normalised_coords = coords.unsqueeze(0) / torch.tensor([H-1, W-1], device=device)
            normalised_coords = 2 * normalised_coords - 1

            # C x H x W
            # Bilinearly sample features[j] at the computed coordinates
            sampled_features = F.grid_sample(features[j].unsqueeze(0), normalised_coords, mode='bilinear', align_corners=True)[0]

            # Display features[i], features[j], sampled_features, and also sampled_features masked by valid_inds
            #  sampled_features : C x H x W, valid_inds : H x W, features[i] : C x H x W

            new_out[i, :, valid_inds] += sampled_features[:, valid_inds]
            counts[i, :, valid_inds] += 1
            agg_means[i, :, valid_inds] += means[j]
            # agg_std[i, :, valid_inds] += stds[j]
            agg_std2[i, :, valid_inds] += stds[j] ** 2

            c = valid_inds.sum().item()
            # if c > 0:
            #     print(f"{i} with {j}:", c, "/", H*W, "=", c/(H*W))
                # debug_display(features, i, j, sampled_features, valid_inds)
                # input()

    new_out = new_out / counts

    # new mean: 1/count (mean1 + mean2 + ...)
    # new std: sqrt(std1^2 + std2^2 + ...) / count
    # goal std: original
    # std_multiplier = agg_std / ((agg_std2 ** 0.5) / counts)
    # current_means = agg_means / counts
    # new_out = (new_out - current_means) * std_multiplier + current_means

    return new_out.to(model_dtype)


def debug_display(features, i, j, sampled_features, valid_inds):
    def show_tensor(t):
        return t[:3].cpu().numpy().transpose(1, 2, 0)

    print("i =", i)
    print("j =", j)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    titles = ['features[i]', 'features[j]', 'sampled_features', 'masked sampled_features']
    images = [
        show_tensor(features[i]),
        show_tensor(features[j]),
        show_tensor(sampled_features),
        show_tensor(valid_inds.unsqueeze(0).float())
    ]

    for ax, img, title in zip(axs, images, titles):
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # normalize to [0,1]
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.show()


# positions3d = get_3d_points(B, 128, 128, device)
# vis(positions3d[:4])
# vis(positions3d)


truncation = 1.0
model_dtype = torch.float16

model = BigGAN.from_pretrained('biggan-deep-256').to(device, dtype=model_dtype)
model.eval()
model.requires_grad_(False)

# for i in range(8):
#     model.generator.layers[i].register_forward_hook(align)
for i in range(2, 8):
    model.generator.layers[i].register_forward_hook(align)


outdir = "out"

N = 256
xs, ys = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
xs, ys = (xs - 0.5)*2, (ys - 0.5)*2  # [-1, 1]
weight = np.minimum(1 - np.abs(xs), 1 - np.abs(ys))
plt.imshow(weight)
plt.show()

for IND in range(32):
    z = torch.from_numpy(truncated_noise_sample(truncation=truncation, batch_size=B)).to(device) * 1
    classes = torch.nn.functional.one_hot(torch.randint(0, 1000, (B,)), num_classes=1000).float().to(device) * 1

    for j in range(1, B):
        classes[j] = classes[0]
        # z[j] = z[0] + torch.randn_like(z[0]) * 0.3

    print(f"Generating {B} images...")
    out = model(z.to(model_dtype), classes.to(model_dtype), truncation)

    # out = B x C x H x W => B x N x 3
    # colors = torch.permute(out, (0, 2, 3, 1)).reshape(B, -1, 3)  # list of point cloud colours
    # positions = [i.reshape(-1, 4) for i in get_3d_points(B, 256, 256, device)]
    # vis(positions, colors)

    print("Panorama computation...")
    numpy_image_arr = [np.array(trf.to_pil_image(out[i].clamp(0, 1))) for i in range(B)]
    pose_arr = [(fov, azimuth, elevation) for azimuth, elevation in zip(azimuths, elevations)]
    width = 2000  # 1152
    mp = MultiPerspectiveWeighted(numpy_image_arr, [weight] * B, pose_arr)
    img = mp.GetEquirec(width // 2, width)
    img = Image.fromarray((np.clip(img, 0, 255)).astype(np.uint8))
    img.save(f"{outdir}/pano_{IND}.png")

    # for i in range(B):
    #     trf.to_pil_image(out[i].clamp(0, 1)).save(f"{outdir}/{IND}_{i}.png")

    gc.collect()

    # out_img = torch.zeros((3, 256, 256 + 128 * (batch_size - 1)), device=out.device, dtype=out.dtype)
    # out_img[:, :, :128] = out[0, :, :, :128]
    # out_img[:, :, -128:] = out[-1, :, :, 128:]
    # for i in range(1, batch_size):
    #     start = 128 * i
    #     end = 128 * i + 128
    #
    #     im1 = out[i-1, :, :, 128:]
    #     im2 = out[i, :, :, :128]
    #     t = torch.linspace(0, 1, 128, device=out.device, dtype=out.dtype)
    #
    #     out_img[:, :, start: end] = im1 + (im2 - im1) * t
    #
    # trf.to_pil_image(out_img.clamp(0, 1)).save(f"{outdir}/{IND}_img.png")




