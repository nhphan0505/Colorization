import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def imshow(L, ab, title):
    img = np.concatenate([L.numpy(), ab.numpy()], axis=1)
    img = img[:32, :, :, :]
    img = np.transpose(img, (0, 2, 3, 1))
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    img = [torch.from_numpy(cv2.cvtColor(img[i], cv2.COLOR_LAB2RGB)).permute(2, 0, 1) for i in range(len(img))] # conver lab to rgb
    img = torch.stack(img, dim=0)
    grid = make_grid(img, nrow=16, padding=2)
    
    plt.figure(figsize=(16,2))
    plt.imshow(grid.permute(1,2,0).numpy()) 
    plt.title(title)
    plt.axis('off')
    plt.show()