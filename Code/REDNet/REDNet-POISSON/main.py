import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
from model import REDNet10, REDNet20, REDNet30
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dataset import BSDDataset
from utils import AverageMeter

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def add_poisson_noise_pytorch(img, scale=1.0, clip=True):
    img = torch.clamp(img, 0, 1)
    noisy = scale*torch.poisson(img/scale)
    if clip:
        noisy = torch.clamp(noisy, 0, 1)
    return noisy


def normalize_patch(patch):
    min_val = patch.min()
    max_val = patch.max()
    return (patch - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(patch)


def calculate_psnr(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    mse = np.mean((pred_np - target_np) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def calculate_ssim(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(target_np, pred_np, data_range=1.0, channel_axis=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet10', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--repeat_factor', type=int, default=100)
    parser.add_argument('--min_var', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--poisson_scale', type=float, default=255.0)
    parser.add_argument('--resume', type=str, default=None)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

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
        # 🔥 Use Poisson NLL Loss
    criterion = nn.PoissonNLLLoss(log_input=False, full=False)
    
    start_epoch = 0
    if opt.resume and os.path.exists(opt.resume):
      checkpoint = torch.load(opt.resume, map_location=device)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      start_epoch = checkpoint['epoch'] + 1
      print(f"✅ Loaded checkpoint from {opt.resume}, resuming at epoch {start_epoch}")

    



    dataset = BSDDataset(
        image_dir=opt.images_dir,
        patch_size=opt.patch_size,
        repeat_factor=opt.repeat_factor,
        min_var=opt.min_var
    )

    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)

    for epoch in range(start_epoch,opt.num_epochs):
        print(f"\nEpoch [{epoch+1}/{opt.num_epochs}] — Total patches: {len(dataset)}")
        epoch_losses = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()

        model.train()
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description(f'Epoch {epoch + 1}/{opt.num_epochs}')
            for clean_inputs, _ in dataloader:
                clean_inputs = clean_inputs.to(device)

                # Step 1: Normalize clean image
                for i in range(clean_inputs.size(0)):
                    clean_inputs[i] = normalize_patch(clean_inputs[i])
                normalized_clean = clean_inputs.clone()

                # Step 2: Add Poisson noise
                noisy_inputs = add_poisson_noise_pytorch(normalized_clean, scale=opt.poisson_scale, clip=True)

                # Step 3: Normalize noisy image
                for i in range(noisy_inputs.size(0)):
                    noisy_inputs[i] = normalize_patch(noisy_inputs[i])

                # Step 4: Model predicts denoised image
                preds = model(noisy_inputs)

                # Step 5: Clamp predictions to avoid log(0)
                preds = torch.clamp(preds, min=1e-6)

                # Step 6: Poisson NLL Loss
                loss = criterion(preds, normalized_clean)

                epoch_losses.update(loss.item(), len(noisy_inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for i in range(preds.size(0)):
                    psnr = calculate_psnr(preds[i], normalized_clean[i])
                    ssim = calculate_ssim(preds[i], normalized_clean[i])
                    psnr_meter.update(psnr)
                    ssim_meter.update(ssim)

                _tqdm.set_postfix(loss=f'{epoch_losses.avg:.6f}', psnr=f'{psnr_meter.avg:.2f}', ssim=f'{ssim_meter.avg:.4f}')
                _tqdm.update(len(noisy_inputs))

        print(f"Epoch {epoch+1} Summary: Loss={epoch_losses.avg:.6f}, PSNR={psnr_meter.avg:.2f} dB, SSIM={ssim_meter.avg:.4f}")

        torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, os.path.join(opt.outputs_dir, f'{opt.arch}_epoch_{epoch+1}.pth'))

        with open(os.path.join(opt.outputs_dir, "metrics_log.csv"), 'a') as f:
            if epoch == 0:
                f.write("epoch,loss,psnr,ssim\n")
            f.write(f"{epoch+1},{epoch_losses.avg:.6f},{psnr_meter.avg:.2f},{ssim_meter.avg:.4f}\n")
