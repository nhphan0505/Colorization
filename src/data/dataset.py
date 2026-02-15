import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from skimage.color import rgb2lab  # pip install scikit-image


class Cifar10LAB(Dataset):
    """
    CIFAR-10 in LAB space

    Input : L  channel tensor (1, H, W) scaled to [0, 1]
    Target: ab channels tensor (2, H, W) scaled to roughly [-1, 1]

    transform: optional applied on the RGB image
               BEFORE converting to LAB, ensuring x/y alignment.
    """

    def __init__(
        self,
        root = "data/raw",
        train = True,
        image_size = 32,
        transform = None,
    ):
        self.transform = transform
        self.image_size = image_size

        self.dataset = CIFAR10(root=root, train=train, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]   # PIL RGB

        # PIL -> numpy (uint8)
        img = np.array(img)

        # Albumentations: numpy -> numpy
        if self.transform is not None:
            img = self.transform(image=img)["image"]

        # numpy RGB [0,255] -> float [0,1]
        rgb = img.astype("float32") / 255.0

        # RGB -> LAB -> split
        lab = rgb2lab(rgb).astype("float32")
        L  = lab[..., 0:1] / 100.0
        ab = lab[..., 1:3] / 128.0

        L  = torch.from_numpy(L).permute(2, 0, 1)
        ab = torch.from_numpy(ab).permute(2, 0, 1)

        return L, ab