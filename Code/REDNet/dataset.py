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
