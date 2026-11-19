import argparse
import os
import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.distributions import Gamma
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model import REDNet10, REDNet20, REDNet30
from dataset import BSDDataset
from utils import AverageMeter

# ---------------- Device ---------------- #
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------- Gamma Noise ---------------- #
def add_multiplicative_gamma_noise(image, k=1.0):
    device = image.device
    concentration = torch.tensor(k, device=device)
    rate = torch.tensor(k, device=device)
    gamma_noise = Gamma(concentration, rate).sample(image.shape)
    noisy_image = image * gamma_noise
    return torch.clamp(noisy_image, 0.0, 1.0)


# ---------------- Normalize ---------------- #
def normalize_patch(patch):
    min_val, max_val = patch.min(), patch.max()
    return (patch - min_val) / (max_val - min_val + 1e-8)
# ------------------ Multiplicative Consistency Loss ------------------ #
class MultiplicativeConsistentLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=self.eps)
        target = torch.clamp(target, min=self.eps)
        log_diff = torch.log(pred) - torch.log(target)
        return torch.mean(log_diff ** 2)

# ---------------- Metrics ---------------- #
def calculate_psnr(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    return peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)

def calculate_ssim(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(target_np, pred_np, data_range=1.0, channel_axis=2)


# ---------------- Main ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="REDNet10", help="REDNet10, REDNet20, REDNet30")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--outputs_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=50)
    parser.add_argument("--repeat_factor", type=int, default=100)
    parser.add_argument("--min_var", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gamma_k_min", type=float, default=2.0, help="Min gamma shape (strong noise)")
    parser.add_argument("--gamma_k_max", type=float, default=20.0, help="Max gamma shape (weak noise)")
    parser.add_argument("--resume", type=str, default=None)
    opt = parser.parse_args()

    # Setup
    os.makedirs(opt.outputs_dir, exist_ok=True)
    torch.manual_seed(opt.seed)

    # Model
    if opt.arch == "REDNet10":
        model = REDNet10()
    elif opt.arch == "REDNet20":
        model = REDNet20()
    elif opt.arch == "REDNet30":
        model = REDNet30()
    else:
        raise ValueError("Invalid REDNet architecture")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = MultiplicativeConsistentLoss()

    # Resume checkpoint
    start_epoch = 0
    if opt.resume and os.path.exists(opt.resume):
        checkpoint = torch.load(opt.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"✅ Loaded checkpoint from {opt.resume}, resuming at epoch {start_epoch}")

    # Dataset
    dataset = BSDDataset(
        image_dir=opt.images_dir,
        patch_size=opt.patch_size,
        repeat_factor=opt.repeat_factor,
        min_var=opt.min_var,
    )
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.threads, pin_memory=True, drop_last=True
    )

    # Training Loop
    for epoch in range(start_epoch, opt.num_epochs):
        print(f"\nEpoch [{epoch+1}/{opt.num_epochs}] — Total patches: {len(dataset)}")
        epoch_losses, psnr_meter, ssim_meter = AverageMeter(), AverageMeter(), AverageMeter()

        model.train()
        with tqdm(total=len(dataset) - len(dataset) % opt.batch_size) as pbar:
            pbar.set_description(f"Epoch {epoch+1}/{opt.num_epochs}")

            for clean_inputs, _ in dataloader:
                clean_inputs = clean_inputs.to(device)
                for i in range(clean_inputs.size(0)):
                    clean_inputs[i] = normalize_patch(clean_inputs[i])

                # Random gamma shape for each image
                k_batch = torch.empty(clean_inputs.size(0)).uniform_(
                    opt.gamma_k_min, opt.gamma_k_max
                ).to(device)

                # Add gamma noise per image
                noisy_inputs = torch.stack([
                    add_multiplicative_gamma_noise(clean_inputs[i], k_batch[i])
                    for i in range(clean_inputs.size(0))
                ]).to(device)

                for i in range(noisy_inputs.size(0)):
                    noisy_inputs[i] = normalize_patch(noisy_inputs[i])

                preds = model(noisy_inputs)
                loss = criterion(preds, clean_inputs)

                epoch_losses.update(loss.item(), len(noisy_inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                for i in range(preds.size(0)):
                    psnr_meter.update(calculate_psnr(preds[i], clean_inputs[i]))
                    ssim_meter.update(calculate_ssim(preds[i], clean_inputs[i]))

                pbar.set_postfix(loss=f"{epoch_losses.avg:.6f}",
                                 psnr=f"{psnr_meter.avg:.2f}",
                                 ssim=f"{ssim_meter.avg:.4f}")
                pbar.update(len(noisy_inputs))

        print(f"Epoch {epoch+1} Summary: Loss={epoch_losses.avg:.6f}, "
              f"PSNR={psnr_meter.avg:.2f} dB, SSIM={ssim_meter.avg:.4f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, os.path.join(opt.outputs_dir, f"{opt.arch}_epoch_{epoch+1}.pth"))

        with open(os.path.join(opt.outputs_dir, "metrics_log.csv"), "a") as f:
            if epoch == start_epoch:
                f.write("epoch,loss,psnr,ssim\n")
            f.write(f"{epoch+1},{epoch_losses.avg:.6f},{psnr_meter.avg:.2f},{ssim_meter.avg:.4f}\n")
