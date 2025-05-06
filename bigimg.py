import torch
from PIL import Image
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
from whatis import whatis as wi
from tqdm import tqdm
import torchvision.transforms.functional as trf
import matplotlib.pyplot as plt


device = torch.device('cuda')
truncation = 1.0

model = BigGAN.from_pretrained('biggan-deep-256').to(device)
model.eval()
model.requires_grad_(False)


@torch.no_grad()
def hook(m, i, out):
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


for i in range(8):
    model.generator.layers[i].register_forward_hook(hook)

batch_size = 6

for IND in range(32):
    z = torch.from_numpy(truncated_noise_sample(truncation=truncation, batch_size=batch_size)).to(device) * 3
    classes = torch.nn.functional.one_hot(torch.randint(0, 1000, (batch_size,)), num_classes=1000).float().to(device) * 2

    for j in range(1, batch_size):
        # classes[j] = classes[0]
        z[j] = z[0]

    print("generating..")
    out = model(z, classes, truncation)
    print("done!")

    outdir = "bigimg_out"

    # for i in range(batch_size):
    #     trf.to_pil_image(out[i].clamp(0, 1)).save(f"{outdir}/{IND}_{i}.png")

    out_img = torch.zeros((3, 256, 256 + 128 * (batch_size - 1)), device=out.device, dtype=out.dtype)
    out_img[:, :, :128] = out[0, :, :, :128]
    out_img[:, :, -128:] = out[-1, :, :, 128:]
    for i in range(1, batch_size):
        start = 128 * i
        end = 128 * i + 128

        im1 = out[i-1, :, :, 128:]
        im2 = out[i, :, :, :128]
        t = torch.linspace(0, 1, 128, device=out.device, dtype=out.dtype)

        out_img[:, :, start: end] = im1 + (im2 - im1) * t

    trf.to_pil_image(out_img.clamp(0, 1)).save(f"{outdir}/{IND}_img.png")
