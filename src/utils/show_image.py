import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension")

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from src.color.lab_to_rgb import lab_to_rgb

def show_img(L, ab, title = "Colorized"):
    img = torch.from_numpy(lab_to_rgb(L, ab)).permute(0,3,1,2)
    grid = make_grid(img, nrow=10, padding=2)
    plt.figure(figsize=(16,2))
    plt.imshow(grid.permute(1,2,0).numpy()) 
    plt.title(title) 
    plt.axis('off')
    plt.show() 