'''import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BSDDataset(Dataset):
    def __init__(self, image_dir, patch_size=50, repeat_factor=100, min_var=0.0):
        self.image_files = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        ])
        self.patch_size = patch_size
        self.repeat_factor = repeat_factor
        self.min_var = min_var
        self.total_patches = len(self.image_files) * repeat_factor

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_files)
        image = Image.open(self.image_files[img_idx]).convert('RGB')
        img_np = np.array(image).astype(np.float32) / 255.0  # normalize [0,1]

        # Random crop
        h, w, _ = img_np.shape
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(f"Image too small: {self.image_files[img_idx]}")

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        patch = img_np[y:y + self.patch_size, x:x + self.patch_size, :]

        # convert to CHW tensor-like numpy
        patch = np.transpose(patch, (2, 0, 1))  # [C,H,W]

        return patch, 0  # second value is dummy (to match DataLoader)

    def __len__(self):
        return self.total_patches
'''
from PIL import Image
import os
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class BSDDataset(Dataset):
    def __init__(self, image_dir, patch_size=50, repeat_factor=100, min_var=0.0):
        self.image_filenames = [
            os.path.join(image_dir, x)
            for x in os.listdir(image_dir)
            if x.endswith('.jpg') or x.endswith('.png')
        ]
        self.patch_size = patch_size
        self.repeat_factor = repeat_factor
        self.min_var = min_var
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames) * self.repeat_factor

    def __getitem__(self, index):
        while True:
            # Random image and patch
            img_path = random.choice(self.image_filenames)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)  # [C, H, W]

            C, H, W = img.shape
            if H < self.patch_size or W < self.patch_size:
                img = transforms.Resize((self.patch_size + 20, self.patch_size + 20))(img)
                C, H, W = img.shape

            top = random.randint(0, H - self.patch_size)
            left = random.randint(0, W - self.patch_size)
            patch = img[:, top:top + self.patch_size, left:left + self.patch_size]

            # Optional: skip smooth patches
            if patch.var().item() >= self.min_var:
                return patch, patch