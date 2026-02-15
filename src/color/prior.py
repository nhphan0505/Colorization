import numpy as np
import torch
from pathlib import Path
from src.color.quantize import soft_encode

ROOT_DIR = Path(__file__).resolve().parents[2]
PTS_PATH = ROOT_DIR / "assets" / "pts_in_hull.npy"
ab_bins = torch.from_numpy(np.load(PTS_PATH, allow_pickle=True)).to(torch.float32)

def color_weight(data, sigma = 5, lam = 0.5):
    #===========Prior==============
    p = torch.zeros(313)
    for _, label in data:
        Z = soft_encode(label)
        p += Z.sum(dim=(0, 2, 3))
    p = p / (p.sum() + 1e-12)

    #=========Smooth prior=========
    c = ab_bins

    d = torch.cdist(c, c)
    K = torch.exp(-(d ** 2) / (2.0 * sigma ** 2))

    p_tilde = K @ p
    p_tilde = p_tilde / (p_tilde.sum() + 1e-12)

    #============Weight============
    Q = p_tilde.numel()

    mixed = (1 - lam) * p_tilde + lam / Q
    w = 1.0 / (mixed + 1e-12)
    w = w / (p_tilde * w).sum()
    torch.save(w, ROOT_DIR / "assets" / "weights.pt")
    return w