import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions import Gamma
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model_gaussian import FFDNet  # your FFDNet implementation
from dataset import BSDDataset
from utils import AverageMeter

# Device
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------- Add Multiplicative Gamma Noise ---------------- #

def add_multiplicative_gamma_noise(image, k=1.0):
    device = image.device
    concentration = torch.tensor(k, device=device)
    rate = torch.tensor(k, device=device)
    gamma_noise = Gamma(concentration, rate).sample(image.shape)
    noisy_image = image * gamma_noise
    return torch.clamp(noisy_image, 0.0, 1.0)

# ---------------- Normalize Patch Min-Max ---------------- #
def normalize_patch(patch):
    min_val = patch.min()
    max_val = patch.max()
    return (patch - min_val) / (max_val - min_val + 1e-8)

# ---------------- Metrics ---------------- #
def calculate_psnr(pred, target):
    return peak_signal_noise_ratio(
        target.squeeze().detach().cpu().numpy(),
        pred.squeeze().detach().cpu().numpy(),
        data_range=1.0
    )

def calculate_ssim(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(target_np, pred_np, data_range=1.0, channel_axis=2)

# ---------------- Checkpoint helpers ---------------- #
def save_checkpoint(epoch, model, optimizer, save_dir):
    path = os.path.join(save_dir, f"FFDNet_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"✅ Saved checkpoint: {path}")
class MultiplicativeConsistentLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=self.eps)
        target = torch.clamp(target, min=self.eps)
        log_diff = torch.log(pred) - torch.log(target)
        return torch.mean(log_diff ** 2)
def load_checkpoint(path, model, optimizer):
    chk = torch.load(path, map_location=device)
    model.load_state_dict(chk['model_state_dict'])
    optimizer.load_state_dict(chk['optimizer_state_dict'])
    start_epoch = chk.get('epoch', 0) + 1
    print(f"✅ Resumed from {path} at epoch {start_epoch}")
    return model, optimizer, start_epoch

# ---------------- Main ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--outputs_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=50)
    parser.add_argument("--repeat_factor", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gamma_k", type=float, default=5.0, help="Gamma shape parameter")
    parser.add_argument("--resume", type=str, default=None)
    opt = parser.parse_args()

    os.makedirs(opt.outputs_dir, exist_ok=True)
    torch.manual_seed(opt.seed)

    # ---------------- Model ---------------- #
    model = FFDNet(is_gray=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = MultiplicativeConsistentLoss()

    start_epoch = 0
    if opt.resume and os.path.exists(opt.resume):
        model, optimizer, start_epoch = load_checkpoint(opt.resume, model, optimizer)

    # ---------------- Dataset ---------------- #
    dataset = BSDDataset(
        image_dir=opt.images_dir,
        patch_size=opt.patch_size,
        repeat_factor=opt.repeat_factor
    )

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                            num_workers=opt.threads, pin_memory=True, drop_last=True)

    # ---------------- Training Loop ---------------- #
    for epoch in range(start_epoch, opt.num_epochs):
        print(f"\nEpoch [{epoch+1}/{opt.num_epochs}] — Total patches: {len(dataset)}")
        epoch_losses = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()

        model.train()
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as pbar:
            pbar.set_description(f"Epoch {epoch+1}/{opt.num_epochs}")

            for clean_inputs, _ in dataloader:
                clean_inputs = clean_inputs.to(device)

                # Normalize clean
                for i in range(clean_inputs.size(0)):
                    clean_inputs[i] = normalize_patch(clean_inputs[i])

                # Add Gamma noise
                noisy_inputs = add_multiplicative_gamma_noise(clean_inputs, k=opt.gamma_k)

                # Normalize noisy
                for i in range(noisy_inputs.size(0)):
                    noisy_inputs[i] = normalize_patch(noisy_inputs[i])

                # Create noise level map (scalar per image)
                N, C, H, W = noisy_inputs.size()
                noise_map = torch.full((N,), 1.0 / opt.gamma_k, device=device)

                # Forward
                preds = model(noisy_inputs, noise_map)
                loss = criterion(preds, clean_inputs)
                epoch_losses.update(loss.item(), len(noisy_inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                for i in range(preds.size(0)):
                    psnr = calculate_psnr(preds[i], clean_inputs[i])
                    ssim = calculate_ssim(preds[i], clean_inputs[i])
                    psnr_meter.update(psnr)
                    ssim_meter.update(ssim)

                pbar.set_postfix(loss=f"{epoch_losses.avg:.6f}",
                                 psnr=f"{psnr_meter.avg:.2f}",
                                 ssim=f"{ssim_meter.avg:.4f}")
                pbar.update(len(noisy_inputs))

        print(f"Epoch {epoch+1}: Loss={epoch_losses.avg:.6f}, "
              f"PSNR={psnr_meter.avg:.2f} dB, SSIM={ssim_meter.avg:.4f}")

        save_checkpoint(epoch, model, optimizer, opt.outputs_dir)

        with open(os.path.join(opt.outputs_dir, "metrics_log.csv"), "a") as f:
            if epoch == start_epoch:
                f.write("epoch,loss,psnr,ssim\n")
            f.write(f"{epoch+1},{epoch_losses.avg:.6f},{psnr_meter.avg:.2f},{ssim_meter.avg:.4f}\n")
