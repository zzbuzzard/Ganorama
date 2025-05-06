import torch
from PIL import Image
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
import torchvision.transforms.functional as trf
import os
from os.path import join

from util import shared_args, device


@torch.no_grad()
def bigimg_hook(_m, _i, out: torch.Tensor) -> torch.Tensor:
    """
    Like with panorama generation, averages each feature across all the other features which it spatially corresponds
    to. This is a lot simpler for flat images than when working on the surface of a sphere.
    """
    b, c, h, w = out.shape
    half = w//2

    # img[i] overlaps 50% with img[i+1] and img[i-1]

    new_out = torch.zeros_like(out)

    for i in range(b):
        if i > 0:
            new_out[i, :, :, :half] = (out[i-1, :, :, half:] + out[i, :, :, :half]) / 2
        else:
            new_out[i, :, :, :half] = out[i, :, :, :half]

        if i < b - 1:
            new_out[i, :, :, half:] = (out[i+1, :, :, :half] + out[i, :, :, half:]) / 2
        else:
            new_out[i, :, :, half:] = out[i, :, :, half:]

    return new_out


if __name__ == "__main__":
    parser = shared_args("Big image generation using BigGAN.")
    parser.add_argument("-o", "--outdir", type=str, default="out/bigimg", help="Directory to save outputs. Will be saved as [ctr].png for the least value of [ctr] which does not exist.")
    parser.add_argument("--batch_size", type=int, default=6, help="Number of BigGAN generations per output image. Output width is (N/2) * (batch_size+1), where N is the model output size.")
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
        model.generator.layers[i].register_forward_hook(bigimg_hook)

    batch_size = args.batch_size

    for repeat in range(args.num_generations):
        z = torch.from_numpy(truncated_noise_sample(truncation=args.truncation, batch_size=batch_size)).to(device)

        if class_names is not None:
            name = class_names[repeat % len(class_names)]
            print(f"[Info] Using class name '{name}' for this bigimg (for all gens)")
            classes = torch.tensor(one_hot_from_names(name, batch_size=batch_size), device=device)
        else:
            classes = torch.nn.functional.one_hot(torch.randint(0, 1000, (batch_size,)), num_classes=model.config.num_classes).float().to(device)

        z = z * args.z_mul
        classes = classes * args.class_mul

        for j in range(1, batch_size):
            if args.share_class: classes[j] = classes[0]
            if args.share_z: z[j] = z[0]

        print(f"[Info] Generating {batch_size} images with {args.biggan_model}... (iteration {repeat+1} / {args.num_generations})")
        out = model(z.to(model_dtype), classes.to(model_dtype), args.truncation)
        print("[Info] Generation complete! ")

        size = model.config.output_dim
        half = size // 2

        # Aggregate into one image, linearly interpolating between each generation to improve cohesion
        out_img = torch.zeros((3, size, size + half * (batch_size - 1)), device=out.device, dtype=out.dtype)
        out_img[:, :, :half] = out[0, :, :, :half]
        out_img[:, :, -half:] = out[-1, :, :, half:]
        for i in range(1, batch_size):
            start = half * i
            end = half * i + half

            im1 = out[i-1, :, :, half:]
            im2 = out[i, :, :, :half]
            t = torch.linspace(0, 1, half, device=out.device, dtype=out.dtype)

            out_img[:, :, start: end] = im1 + (im2 - im1) * t

        save_path = join(args.outdir, f"{ctr}.png")
        print(f"[Info] Saving to {save_path}")
        trf.to_pil_image(out_img.clamp(0, 1)).save(save_path)
        ctr += 1
