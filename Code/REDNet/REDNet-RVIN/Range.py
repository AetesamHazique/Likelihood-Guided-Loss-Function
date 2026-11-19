import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import REDNet10, REDNet20, REDNet30
from dataset import BSDDataset
from utils import AverageMeter

# ------------------ Setup ------------------ #
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ------------------ Add RVIN Noise ------------------ #
def add_rvin(image, density=0.2, value_range=(0., 1.)):
    noisy = image.clone()
    mask = torch.rand_like(noisy[0]) < density  # choose random pixels (per channel)
    noise_vals = torch.empty_like(noisy).uniform_(*value_range)
    noisy[:, mask] = noise_vals[:, mask]
    return noisy


# ------------------ Normalize Min-Max ------------------ #
def normalize_patch(patch):
    min_val, max_val = patch.min(), patch.max()
    return (patch - min_val) / (max_val - min_val + 1e-8)


# ------------------ Adaptive Laplace Loss ------------------ #
class AdaptiveLaplaceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_b = nn.Parameter(torch.zeros(1))  # Learnable scale

    def forward(self, y_true, y_pred):
        b = torch.exp(self.log_b).to(y_true.device)
        log_b = self.log_b.to(y_true.device)
        residual = torch.abs(y_true - y_pred)
        return torch.mean(residual / b + log_b)


# ------------------ Metrics ------------------ #
def calculate_psnr(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    return peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)


def calculate_ssim(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(target_np, pred_np, data_range=1.0, channel_axis=2)


# ------------------ Main ------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet10', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--repeat_factor', type=int, default=100)
    parser.add_argument('--min_var', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)

    # RVIN noise range
    parser.add_argument('--rvin_min', type=float, default=0.05, help='Minimum RVIN density')
    parser.add_argument('--rvin_max', type=float, default=0.3, help='Maximum RVIN density')

    # Resume checkpoint
    parser.add_argument('--resume', type=str, default=None)

    opt = parser.parse_args()

    # ------------------ Prepare Environment ------------------ #
    os.makedirs(opt.outputs_dir, exist_ok=True)
    torch.manual_seed(opt.seed)

    # ------------------ Model ------------------ #
    if opt.arch == 'REDNet10':
        model = REDNet10()
    elif opt.arch == 'REDNet20':
        model = REDNet20()
    elif opt.arch == 'REDNet30':
        model = REDNet30()
    else:
        raise ValueError("Invalid architecture name")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = AdaptiveLaplaceLoss().to(device)

    start_epoch = 0

    # ------------------ Resume Checkpoint ------------------ #
    if opt.resume and os.path.exists(opt.resume):
        checkpoint = torch.load(opt.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Loaded checkpoint from {opt.resume}, resuming at epoch {start_epoch}")

    # ------------------ Dataset ------------------ #
    dataset = BSDDataset(
        image_dir=opt.images_dir,
        patch_size=opt.patch_size,
        repeat_factor=opt.repeat_factor,
        min_var=opt.min_var,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.threads,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------ Training Loop ------------------ #
    for epoch in range(start_epoch, opt.num_epochs):
        print(f"\nEpoch [{epoch+1}/{opt.num_epochs}] — Total patches: {len(dataset)}")
        epoch_losses, psnr_meter, ssim_meter = AverageMeter(), AverageMeter(), AverageMeter()

        model.train()
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as pbar:
            pbar.set_description(f'Epoch {epoch + 1}/{opt.num_epochs}')

            for clean_inputs, _ in dataloader:
                clean_inputs = clean_inputs.to(device)

                # Normalize clean patches
                clean_inputs = torch.stack([normalize_patch(p) for p in clean_inputs])

                # Random RVIN density per image
                rvin_densities = torch.empty(clean_inputs.size(0)).uniform_(
                    opt.rvin_min, opt.rvin_max
                ).to(device)

                noisy_inputs = torch.stack([
                    add_rvin(clean_inputs[i], density=rvin_densities[i].item())
                    for i in range(clean_inputs.size(0))
                ]).to(device)

                noisy_inputs = torch.stack([normalize_patch(p) for p in noisy_inputs])

                preds = model(noisy_inputs)
                loss = criterion(clean_inputs, preds)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.update(loss.item(), len(noisy_inputs))

                # Metrics
                for i in range(preds.size(0)):
                    psnr = calculate_psnr(preds[i], clean_inputs[i])
                    ssim = calculate_ssim(preds[i], clean_inputs[i])
                    psnr_meter.update(psnr)
                    ssim_meter.update(ssim)

                pbar.set_postfix(loss=f'{epoch_losses.avg:.6f}',
                                 psnr=f'{psnr_meter.avg:.2f}',
                                 ssim=f'{ssim_meter.avg:.4f}')
                pbar.update(len(noisy_inputs))

        print(f"Epoch {epoch+1} Summary: Loss={epoch_losses.avg:.6f}, "
              f"PSNR={psnr_meter.avg:.2f} dB, SSIM={ssim_meter.avg:.4f}")

        # ------------------ Save Checkpoint ------------------ #
        checkpoint_path = os.path.join(opt.outputs_dir, f'{opt.arch}_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")

        # ------------------ Log Metrics ------------------ #
        log_path = os.path.join(opt.outputs_dir, "metrics_log.csv")
        with open(log_path, 'a') as f:
            if epoch == start_epoch:
                f.write("epoch,loss,psnr,ssim\n")
            f.write(f"{epoch+1},{epoch_losses.avg:.6f},{psnr_meter.avg:.2f},{ssim_meter.avg:.4f}\n")
