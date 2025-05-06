import torch
import math
from typing import List
import random
import matplotlib.pyplot as plt
try:
    import open3d as o3d
except ImportError:
    print("[Warning] Could not import open3d, 3D visualisation will not work.")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_intrinsic(fov: float, H: int, W: int) -> torch.Tensor:
    """Get 3x4 intrinsic camera matrix"""
    f = (W / 2) / math.tan(math.radians(fov) / 2)
    cx, cy = (W-1) / 2, (H-1) / 2
    K = torch.tensor([
        [f, 0, cx, 0],
        [0, f, cy, 0],
        [0, 0,  1, 0],
    ], dtype=torch.float32, device=device)
    return K


def get_extrinsic(azimuth: float, elevation: float) -> torch.Tensor:
    """Get 4x4 extrinsic matrix for this azimuth (left-right) and elevation angle. Translation fixed to 0."""
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


def visualise_point_clouds(positions3d: List[torch.Tensor], color_list: List[torch.Tensor] = None):
    """Visualise a series of point clouds, with optional colours. Mainly useful for debugging."""
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


def plt_display(features, i, j, sampled_features, valid_inds):
    """Visualise the sampling operation"""
    def show_tensor(t):
        return t[:3].cpu().numpy().transpose(1, 2, 0)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    titles = [f'features[{i}]', f'features[{j}]', 'sampled_features', 'sampled_features mask']
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

