import torch
from PIL import Image
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
import torchvision.transforms.functional as trf
import numpy as np
import math
import torch.nn.functional as F
import gc
import argparse
import os
from os.path import join

from util import shared_args, get_extrinsic, get_intrinsic, plt_display, visualise_point_clouds, device
from pers2equir import MultiPerspectiveWeighted

fov = 90

# Four angles looking up and down, eight looking directly straight
azimuths = [i * 90 for i in range(4)] * 2 + [i * 45 for i in range(8)]
elevations = [45] * 4 + [-45] * 4 + [0] * 8

B = len(azimuths)  # Effective batch size (by default, 4 + 4 + 8 = 16)

# B x 4 x 4
poses = [get_extrinsic(math.radians(a), math.radians(e)) for a, e in zip(azimuths, elevations)]
inverse_poses = [torch.linalg.inv(M) for M in poses]


# Returns tensor of shape (B x H x W x 4), giving the 3D positions of each feature for each camera
@torch.no_grad()
def get_3d_points(H: int, W: int) -> torch.Tensor:
    """Compute 3D positions of all features for all B cameras. Output shape (B x H x W x 4)."""
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
def enforce_view_consistency(_m, _i, features: torch.Tensor) -> torch.Tensor:
    """
    Given features of shape (B x C x H x W), with corresponding camera features in `poses`,
     enforces view consistency by finding and averaging corresponding areas of the views.
    This function is run as a PyTorch forward hook on several internal layers of the GAN,
     ensuring multi-view consistency at intermediate stages.
    """
    features = features.to(torch.float32)  # model may be in fp16, but carry these computations out in fp32

    B, C, H, W = features.shape
    device = features.device

    K = get_intrinsic(fov, H, W)               # 3 x 4 intrinsic matrix
    Ps = [K @ pose for pose in inverse_poses]  # List of 3 x 4 - complete camera matrix

    # Compute 3D positions of all features
    positions3d = get_3d_points(H, W)

    # The output will be aggregated in new_out via sum, also tracking per-pixel counts to allow mean
    new_out = features.clone()
    counts = torch.ones((B, 1, H, W), device=device)

    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            # Project positions3d[i] into view j, to find pixels in view j which correspond to our pixels
            # (H x W x 4) -> (4 x N) -> (3 x N)
            uvw = Ps[j] @ positions3d[i].reshape(-1, 4).t()
            # Normalise to pixel coords (2 x N)
            uv = uvw[:2] / uvw[2:]

            # Compute which pixels are also visible in j's camera (H x W bool map)
            depth = uvw.t().reshape(H, W, 3)[..., 2]
            coords = uv.t().reshape(H, W, 2)
            valid_inds = (coords[..., 0] >= 0) & (coords[..., 0] <= H-1) & (coords[..., 1] >= 0) & (coords[..., 1] <= W-1)
            valid_inds = valid_inds & (depth > 0)

            # Prepare normalised_coords for F.grid_sample: normalise [0, H-1] to [-1, 1]
            normalised_coords = coords.unsqueeze(0) / torch.tensor([H-1, W-1], device=device)
            normalised_coords = 2 * normalised_coords - 1

            # Bilinearly sample features[j] at the corresponding coordinates (C x H x W output)
            sampled_features = F.grid_sample(features[j].unsqueeze(0), normalised_coords, mode='bilinear', align_corners=True)[0]

            # Display features[i], features[j], sampled_features, and also sampled_features masked by valid_inds
            #  sampled_features : C x H x W, valid_inds : H x W, features[i] : C x H x W

            # Add the sampled features from j to the corresponding points in i
            new_out[i, :, valid_inds] += sampled_features[:, valid_inds]
            counts[i, :, valid_inds] += 1

            # For creating assets/explain.png
            # c = valid_inds.sum().item()
            # if c > 0 and H >= 128:
            #     print(f"{i} with {j}:", c, "/", H*W, "=", c/(H*W))
            #     plt_display(features, i, j, sampled_features, valid_inds)
            #     input()

    # Divide by count to average each pixel after sum aggregation
    new_out = new_out / counts

    # Convert intermediate result back to model dtype
    return new_out.to(model_dtype)


if __name__ == "__main__":
    parser = shared_args("Panorama generation using BigGAN.")

    parser.add_argument("-o", "--outdir", type=str, default="out/panorama", help="Directory to save outputs. Will be saved as [ctr].png for the least value of [ctr] which does not exist.")
    parser.add_argument("--panorama_width", type=int, default=2000)
    parser.add_argument("--panorama_height", type=int, default=1000)
    parser.add_argument("--visualise_point_cloud", action="store_true", help="Visualise panorama as a point cloud, with overlaps from the multiple generations. Requires open3d to be installed.")

    args = parser.parse_args()

    class_names = args.class_names.split("|") if args.class_names is not None else None

    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.float32

    print(f"[Info] Loading {args.biggan_model} with dtype = {args.model_dtype}")
    model = BigGAN.from_pretrained(args.biggan_model).to(device, dtype=model_dtype)
    model.eval()
    model.requires_grad_(False)

    ctr = 0
    while os.path.exists(join(args.outdir, f"{ctr}.png")):
        ctr += 1

    for i in range(args.start_layer, args.end_layer + 1):
        model.generator.layers[i].register_forward_hook(enforce_view_consistency)

    # Prepare weight mask to lerp between 3D views on the panorama
    #  (note: not necessary if all layers have view consistency, but improves quality if only some layers are enforced)
    size = model.config.output_dim
    xs, ys = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    xs, ys = (xs - 0.5)*2, (ys - 0.5)*2  # [-1, 1]
    weight = np.minimum(1 - np.abs(xs), 1 - np.abs(ys))

    for repeat in range(args.num_generations):
        z = torch.from_numpy(truncated_noise_sample(truncation=args.truncation, batch_size=B)).to(device)

        if class_names is not None:
            name = class_names[repeat % len(class_names)]
            print(f"[Info] Using class name '{name}' for this panorama (for all gens)")
            classes = torch.tensor(one_hot_from_names(name, batch_size=B), device=device)
        else:
            classes = torch.nn.functional.one_hot(torch.randint(0, 1000, (B,)), num_classes=model.config.num_classes).float().to(device)

        z = z * args.z_mul
        classes = classes * args.class_mul

        for j in range(1, B):
            if args.share_class: classes[j] = classes[0]
            if args.share_z: z[j] = z[0]

        print(f"[Info] Generating {B} images with {args.biggan_model}... (iteration {repeat+1} / {args.num_generations})")
        out = model(z.to(model_dtype), classes.to(model_dtype), args.truncation)
        print("[Info] Generation complete! ")

        print("[Info] Computing panorama...")
        numpy_image_arr = [np.array(trf.to_pil_image(out[i].clamp(0, 1))) for i in range(B)]
        pose_arr = [(fov, azimuth, elevation) for azimuth, elevation in zip(azimuths, elevations)]
        mp = MultiPerspectiveWeighted(numpy_image_arr, [weight] * B, pose_arr)
        img = mp.GetEquirec(args.panorama_height, args.panorama_width)

        save_path = join(args.outdir, f"{ctr}.png")
        print(f"[Info] Saving to {save_path}")
        img = Image.fromarray((np.clip(img, 0, 255)).astype(np.uint8))
        img.save(save_path)
        ctr += 1

        if args.visualise_point_cloud:
            print("[Info] Visualising point cloud...")
            # Reshape Bx3xHxW -> BxNx3
            colors = torch.permute(out, (0, 2, 3, 1)).reshape(B, -1, 3)  # list of point cloud colours
            # Reshape positions to Nx4 for each
            positions = [i.reshape(-1, 4) for i in get_3d_points(size, size)]
            visualise_point_clouds(positions, colors)

        gc.collect()
