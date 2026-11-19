
import os
import torch
import numpy as np
from PIL import Image
from torch import nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import DnCNN
from model_gaussian import FFDNet
# If REDNet is in a separate file, import it here:
# from REDNet_model import REDNet


# =====================================================
#                  CONFIGURATION
# =====================================================

MODEL_TYPE = "DnCNN"       # "DnCNN", "FFDNet", "REDNet"

NOISE_TYPE = "gaussian"    # gaussian, poisson, rvin, gamma

CHECKPOINT_PATH = "/content/drive/MyDrive/checkpoints/model.pth"

TEST_IMAGES_DIR = "/content/drive/MyDrive/BSD500/test"
OUTPUT_DIR = "/content/drive/MyDrive/Testing_Results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
DNCCN_LAYERS = 17

# Noise levels
GAUSSIAN_STD = 25 / 255.0
POISSON_SCALE = 2.0
RVIN_DENSITY = 0.1
GAMMA_K = 5.0

# For range-trained models, choose a value inside the
# training range for each test image.
USE_RANGE = False

GAUSSIAN_MIN = 5 / 255.0
GAUSSIAN_MAX = 50 / 255.0

POISSON_MIN = 1.0
POISSON_MAX = 5.0

RVIN_MIN = 0.05
RVIN_MAX = 0.40

GAMMA_K_MIN = 2.0
GAMMA_K_MAX = 15.0


# =====================================================
#                  MODEL LOADING
# =====================================================

def load_model():

    if MODEL_TYPE == "DnCNN":
        model = DnCNN(num_layers=DNCCN_LAYERS)

    elif MODEL_TYPE == "FFDNet":
        model = FFDNet(is_gray=False)

    elif MODEL_TYPE == "REDNet":
        # Uncomment and use your actual REDNet class.
        raise NotImplementedError(
            "Import and initialize your REDNet architecture here."
        )

    else:
        raise ValueError("Unknown model type")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded:", MODEL_TYPE)

    return model


# =====================================================
#                  NORMALIZATION
# =====================================================

def normalize_patch(image):

    min_val = image.min()
    max_val = image.max()

    return (image - min_val) / (max_val - min_val + 1e-8)


# =====================================================
#                  NOISE SIMULATION
# =====================================================

def add_gaussian(image, std):

    noise = torch.randn_like(image) * std

    return torch.clamp(image + noise, 0.0, 1.0)


def add_poisson(image, scale):

    image = torch.clamp(image, 0, 1)

    noisy = scale * torch.poisson(image / scale)

    return torch.clamp(noisy, 0, 1)


def add_rvin(image, density):

    noisy = image.clone()

    mask = torch.rand_like(noisy[0]) < density

    noise_values = torch.empty_like(noisy).uniform_(0, 1)

    noisy[:, mask] = noise_values[:, mask]

    return noisy


def add_gamma(image, k):

    concentration = torch.tensor(k, device=image.device)
    rate = torch.tensor(k, device=image.device)

    gamma_noise = torch.distributions.Gamma(
        concentration, rate
    ).sample(image.shape)

    noisy = image * gamma_noise

    return torch.clamp(noisy, 0, 1)


# =====================================================
#                  METRICS
# =====================================================

def calculate_psnr(pred, target):

    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.detach().cpu().numpy().transpose(1, 2, 0)

    return peak_signal_noise_ratio(
        target_np, pred_np, data_range=1.0
    )


def calculate_ssim(pred, target):

    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.detach().cpu().numpy().transpose(1, 2, 0)

    return structural_similarity(
        target_np,
        pred_np,
        data_range=1.0,
        channel_axis=2
    )


# =====================================================
#                  SAVE IMAGE
# =====================================================

def save_image(tensor, path):

    image = tensor.detach().cpu().clamp(0, 1)

    image = image.permute(1, 2, 0).numpy()

    image = (image * 255).round().astype(np.uint8)

    Image.fromarray(image).save(path)


# =====================================================
#                  TESTING
# =====================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model()

    image_files = [
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
    ]

    image_files.sort()

    psnr_values = []
    ssim_values = []

    print("\nTesting started...")
    print("Model:", MODEL_TYPE)
    print("Noise:", NOISE_TYPE)
    print("Number of images:", len(image_files))

    for index, filename in enumerate(image_files):

        image_path = os.path.join(TEST_IMAGES_DIR, filename)

        # Load clean image
        clean = Image.open(image_path).convert("RGB")

        clean = np.asarray(clean).astype(np.float32) / 255.0

        clean = torch.from_numpy(clean).permute(2, 0, 1).float()

        clean = clean.unsqueeze(0).to(DEVICE)

        # Normalize clean image
        clean = normalize_patch(clean[0]).unsqueeze(0)

        # -------------------------------------------------
        # Select noise level
        # -------------------------------------------------

        if USE_RANGE:

            if NOISE_TYPE == "gaussian":
                noise_level = torch.empty(1).uniform_(
                    GAUSSIAN_MIN, GAUSSIAN_MAX
                ).item()

            elif NOISE_TYPE == "poisson":
                noise_level = torch.empty(1).uniform_(
                    POISSON_MIN, POISSON_MAX
                ).item()

            elif NOISE_TYPE == "rvin":
                noise_level = torch.empty(1).uniform_(
                    RVIN_MIN, RVIN_MAX
                ).item()

            elif NOISE_TYPE == "gamma":
                noise_level = torch.empty(1).uniform_(
                    GAMMA_K_MIN, GAMMA_K_MAX
                ).item()

        else:

            if NOISE_TYPE == "gaussian":
                noise_level = GAUSSIAN_STD

            elif NOISE_TYPE == "poisson":
                noise_level = POISSON_SCALE

            elif NOISE_TYPE == "rvin":
                noise_level = RVIN_DENSITY

            elif NOISE_TYPE == "gamma":
                noise_level = GAMMA_K

            else:
                raise ValueError("Unknown noise type")

        # -------------------------------------------------
        # Add noise
        # -------------------------------------------------

        if NOISE_TYPE == "gaussian":
            noisy = add_gaussian(clean[0], noise_level).unsqueeze(0)

        elif NOISE_TYPE == "poisson":
            noisy = add_poisson(clean[0], noise_level).unsqueeze(0)

        elif NOISE_TYPE == "rvin":
            noisy = add_rvin(clean[0], noise_level).unsqueeze(0)

        elif NOISE_TYPE == "gamma":
            noisy = add_gamma(clean[0], noise_level).unsqueeze(0)

        else:
            raise ValueError("Unsupported noise type")

        # Same noisy-image normalization used in training
        noisy = normalize_patch(noisy[0]).unsqueeze(0)

        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        with torch.no_grad():

            if MODEL_TYPE == "FFDNet":

                if NOISE_TYPE == "gamma":
                    noise_map = torch.tensor(
                        [1.0 / noise_level], device=DEVICE
                    )

                elif NOISE_TYPE == "gaussian":
                    noise_map = torch.tensor(
                        [noise_level], device=DEVICE
                    )

                else:
                    raise ValueError(
                        "Check the noise-map convention used "
                        "when training FFDNet for this noise."
                    )

                restored = model(noisy, noise_map)

            else:
                restored = model(noisy)

        restored = torch.clamp(restored, 0, 1)

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        psnr = calculate_psnr(restored[0], clean[0])
        ssim = calculate_ssim(restored[0], clean[0])

        psnr_values.append(psnr)
        ssim_values.append(ssim)

        # -------------------------------------------------
        # Save results
        # -------------------------------------------------

        name = os.path.splitext(filename)[0]

        save_image(
            clean[0],
            os.path.join(OUTPUT_DIR, name + "_clean.png")
        )

        save_image(
            noisy[0],
            os.path.join(OUTPUT_DIR, name + "_noisy.png")
        )

        save_image(
            restored[0],
            os.path.join(OUTPUT_DIR, name + "_restored.png")
        )

        print(
            f"[{index+1}/{len(image_files)}] {filename} | "
            f"Level={noise_level:.5f} | "
            f"PSNR={psnr:.2f} | SSIM={ssim:.4f}"
        )

    # -----------------------------------------------------
    # Average metrics
    # -----------------------------------------------------

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print("\n========== TEST RESULTS ==========")
    print("Model:", MODEL_TYPE)
    print("Noise:", NOISE_TYPE)
    print("Average PSNR:", round(avg_psnr, 4))
    print("Average SSIM:", round(avg_ssim, 4))
    print("==================================")

    with open(os.path.join(OUTPUT_DIR, "results.txt"), "a") as f:

        f.write(
            f"Model={MODEL_TYPE}, "
            f"Noise={NOISE_TYPE}, "
            f"Range={USE_RANGE}, "
            f"PSNR={avg_psnr:.4f}, "
            f"SSIM={avg_ssim:.4f}\n"
        )


if __name__ == "__main__":
    main()
