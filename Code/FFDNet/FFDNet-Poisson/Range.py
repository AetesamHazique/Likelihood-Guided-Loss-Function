import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import FFDNet
from dataset import BSDDataset
from utils import AverageMeter

# ---------------- Setup ---------------- #
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------- Poisson Noise Addition ---------------- #
def add_poisson_noise_pytorch(img, scale=1.0, clip=True):
    img = torch.clamp(img, 0, 1)
    noisy = scale*torch.poisson(img/scale)
    if clip:
        noisy = torch.clamp(noisy, 0, 1)
    return noisy

# ---------------- Normalize Patch (Min-Max) ---------------- #
def normalize_patch(patch):
    min_val = patch.min()
    max_val = patch.max()
    return (patch - min_val) / (max_val - min_val + 1e-8)


# ---------------- Metrics ---------------- #
def calculate_psnr(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    return peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)


def calculate_ssim(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(target_np, pred_np, data_range=1.0, channel_axis=2)


# ---------------- Checkpoint helpers ---------------- #
def save_checkpoint(epoch, model, optimizer, save_dir, arch):
    model_out_path = os.path.join(save_dir, f"{arch}_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_out_path)
    print(f"✅ Checkpoint saved to {model_out_path}")


def load_checkpoint(path, model, optimizer):
    chk = torch.load(path, map_location=device)
    model.load_state_dict(chk['model_state_dict'])
    optimizer.load_state_dict(chk['optimizer_state_dict'])
    start_epoch = chk.get('epoch', 0) + 1
    print(f"✅ Loaded checkpoint from {path}, resuming at epoch {start_epoch}")
    return model, optimizer, start_epoch


# ---------------- Main ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--repeat_factor', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)

    # ✅ Range of Poisson noise levels
    parser.add_argument('--scale_min', type=float, default=30.0, help="Minimum Poisson noise scale")
    parser.add_argument('--scale_max', type=float, default=255.0, help="Maximum Poisson noise scale")

    parser.add_argument('--resume', type=str, default=None)
    opt = parser.parse_args()

    os.makedirs(opt.outputs_dir, exist_ok=True)
    torch.manual_seed(opt.seed)

    # ---------------- Model ---------------- #
    model = FFDNet(is_gray=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.PoissonNLLLoss(log_input=False, full=False)

    # Resume
    start_epoch = 0
    if opt.resume and os.path.exists(opt.resume):
        model, optimizer, start_epoch = load_checkpoint(opt.resume, model, optimizer)

    # Dataset
    dataset = BSDDataset(
        image_dir=opt.images_dir,
        patch_size=opt.patch_size,
        repeat_factor=opt.repeat_factor
    )

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                            num_workers=opt.threads, pin_memory=True, drop_last=True)

    # ---------------- Training loop ---------------- #
    for epoch in range(start_epoch, opt.num_epochs):
        print(f"\nEpoch [{epoch+1}/{opt.num_epochs}] — Total patches: {len(dataset)}")
        epoch_losses = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()

        model.train()
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description(f'Epoch {epoch+1}/{opt.num_epochs}')

            for clean_inputs, _ in dataloader:
                clean_inputs = clean_inputs.to(device)

                # Normalize clean inputs
                clean_inputs = torch.stack([normalize_patch(img) for img in clean_inputs])

                # ✅ Random Poisson scale per sample (range [scale_min, scale_max])
                scales = torch.empty(clean_inputs.size(0)).uniform_(
                    opt.scale_min, opt.scale_max
                ).to(device)

                # Apply Poisson noise with individual scales
                noisy_inputs = torch.stack([
                    add_poisson_noise_pytorch(clean_inputs[i], scale=scales[i])
                    for i in range(clean_inputs.size(0))
                ])
                for i in range(noisy_inputs.size(0)):
                    noisy_inputs[i] = normalize_patch(noisy_inputs[i])

                # Compute per-sample "sigma" proxy (variance-based)
                sigma_map = torch.sqrt(torch.clamp(clean_inputs, min=1e-8))
                sigma_scalar = sigma_map.view(sigma_map.size(0), -1).mean(dim=1)

                preds = model(noisy_inputs, sigma_scalar)
                preds = torch.clamp(preds, 0, 1)
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

                _tqdm.set_postfix(loss=f'{epoch_losses.avg:.6f}',
                                  psnr=f'{psnr_meter.avg:.2f}',
                                  ssim=f'{ssim_meter.avg:.4f}')
                _tqdm.update(len(noisy_inputs))

        print(f"Epoch {epoch+1} Summary: "
              f"Loss={epoch_losses.avg:.6f}, "
              f"PSNR={psnr_meter.avg:.2f} dB, "
              f"SSIM={ssim_meter.avg:.4f}")

        save_checkpoint(epoch, model, optimizer, opt.outputs_dir, "FFDNet_PoissonRange")

        with open(os.path.join(opt.outputs_dir, "metrics_log.csv"), 'a') as f:
            if epoch == start_epoch:
                f.write("epoch,loss,psnr,ssim\n")
            f.write(f"{epoch+1},{epoch_losses.avg:.6f},"
                    f"{psnr_meter.avg:.2f},{ssim_meter.avg:.4f}\n")
