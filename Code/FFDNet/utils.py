import numpy as np
import cv2
import math
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.util import random_noise


def is_image_gray(image):
    """
    Check if an image is grayscale
    :param image: cv2 image
    """
    return not(len(image.shape) == 3 and not(np.allclose(image[...,0], image[...,1]) and np.allclose(image[...,2], image[...,1])))


def downsample(x):
    """
    Downsample input tensor by factor of 2
    :param x: (N, C, W, H)
    :return: (N, 4*C, W/2, H/2)
    """
    N, C, W, H = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Cout = 4 * C
    Wout = W // 2
    Hout = H // 2

    down_features = torch.zeros((N, Cout, Wout, Hout), device=x.device, dtype=x.dtype)
    
    for idx in range(4):
        down_features[:, idx:Cout:4, :, :] = x[:, :, idxL[idx][0]::2, idxL[idx][1]::2]

    return down_features


def upsample(x):
    """
    Upsample tensor by factor of 2
    :param x: (N, C, W, H)
    :return: (N, C/4, W*2, H*2)
    """
    N, Cin, Win, Hin = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    Cout = Cin // 4
    Wout = Win * 2
    Hout = Hin * 2

    up_feature = torch.zeros((N, Cout, Wout, Hout), device=x.device, dtype=x.dtype)
    for idx in range(4):
        up_feature[:, :, idxL[idx][0]::2, idxL[idx][1]::2] = x[:, idx:Cin:4, :, :]

    return up_feature


def normalize(data):
    """
    Normalize image data to [0,1]
    """
    return np.float32(data / 255)


def image_to_patches(image, patch_size):
    """
    Split image into patches
    :param image: (C, W, H) numpy
    :param patch_size: int
    :return: (num_patches, C, patch_size, patch_size)
    """
    W = image.shape[1]
    H = image.shape[2]
    if W < patch_size or H < patch_size:
        return []

    patches = []
    for ws in range(0, W // patch_size):
        for hs in range(0, H // patch_size):
            patch = image[:, ws*patch_size:(ws+1)*patch_size, hs*patch_size:(hs+1)*patch_size]
            patches.append(patch)
    return np.array(patches, dtype=np.float32)


def add_batch_noise(images, noise_sigma):
    """
    Add Gaussian noise to a batch
    :param images: Tensor (N, C, W, H)
    :param noise_sigma: variance of noise
    :return: noisy images (Tensor)
    """
    images_np = images.cpu().numpy()
    noisy = random_noise(images_np, mode='gaussian', var=noise_sigma)
    return torch.FloatTensor(noisy).to(images.device)


def batch_psnr(img, imclean, data_range=1.0):
    """
    Compute PSNR over a batch of images
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i], img_cpu[i], data_range=data_range)
    return psnr / img_cpu.shape[0]


def variable_to_cv2_image(varim):
    """
    Convert PyTorch Variable to CV2 image
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :] * 255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        res = (res*255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res


def weights_init_kaiming(lyr):
    """
    He (Kaiming) initialization
    """
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(lyr.bias.data, 0.0)
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
