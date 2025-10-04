import torch
import cv2
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

batch_size = 256

class dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data=data
        self.transform=transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, _ = self.data[idx]
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        if self.transform:
            img = self.transform(img)
            L = img[0:1, :, :]
            ab = (img[1:3, :, :]*255 - 128) / 128
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float()
            L = img[0:1, :, :] / 255
            ab = (img[1:3, :, :]- 128) / 128
        return L, ab