import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image
import numpy as np
from model import REDNet10, REDNet20, REDNet30

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize_patch(patch):
    min_val = patch.min()
    max_val = patch.max()
    return (patch - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(patch)


def add_gaussian_noise(img, mean=0.0, std=0.1):
    noise = torch.randn_like(img) * std + mean
    return torch.clamp(img + noise, 0., 1.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet10', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--gaussian_std', type=float, default=25, help='Gaussian noise std in 0-255 range')
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    # Load model
    if opt.arch == 'REDNet10':
        model = REDNet10()
    elif opt.arch == 'REDNet20':
        model = REDNet20()
    elif opt.arch == 'REDNet30':
        model = REDNet30()
    else:
        raise ValueError("Invalid architecture name")

    state_dict = torch.load(opt.weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load image
    input_image = Image.open(opt.image_path).convert('RGB')
    filename = os.path.splitext(os.path.basename(opt.image_path))[0]

    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(input_image).unsqueeze(0).to(device)  # [1, C, H, W]

    # Normalize clean image
    image_tensor[0] = normalize_patch(image_tensor[0])

    # Add Gaussian noise
    noisy_tensor = add_gaussian_noise(image_tensor[0], std=opt.gaussian_std / 255.0).unsqueeze(0)

    # Normalize noisy image
    noisy_tensor[0] = normalize_patch(noisy_tensor[0])

    # Save noisy image
    noisy_np = (noisy_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(noisy_np).save(os.path.join(opt.outputs_dir, f"{filename}_noisy.png"))

    # Denoise
    with torch.no_grad():
        output = model(noisy_tensor)

    # Save denoised image
    output_np = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(output_np).save(os.path.join(opt.outputs_dir, f"{filename}_{opt.arch}_denoised.png"))

    print(f"✅ Noisy and denoised images saved to {opt.outputs_dir}")
