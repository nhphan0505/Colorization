import numpy as np
import torch
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]   # Colorization/
PTS_PATH = ROOT_DIR / "assets" / "pts_in_hull.npy"

def soft_encode(ab, topk=5, sigma=5.0):
    ab_bins = torch.from_numpy(np.load(PTS_PATH, allow_pickle=True)).to(torch.float32).to(ab.device)
    ab = ab * 128
    B, _, H, W = ab.shape
    Q = ab_bins.shape[0]

    ab_flat = ab.permute(0, 2, 3, 1).contiguous().view(B, H * W, 2)

    dists = torch.cdist(ab_flat, ab_bins)    
    vals, idxs = torch.topk(dists, k=topk, largest=False, dim=2)

    w = torch.exp(-(vals ** 2) / (2.0 * sigma ** 2))
    w = torch.softmax(w, dim=2)

    Z = torch.zeros((B, H * W, Q)).to(ab.device)
    Z.scatter_(2, idxs, w)

    Z = Z.view(B, H, W, Q).permute(0, 3, 1, 2).contiguous()
    return Z

def soft_decode(z, T, logits = True):
    ab_bins = torch.from_numpy(np.load(PTS_PATH, allow_pickle=True)).to(torch.float32).to(z.device)
    if logits:
        zT = torch.softmax(z/T, dim=1)
    else:
        zT = torch.softmax(torch.log(z + 1e-8)/T, dim=1)
    a = (zT * ab_bins[:,0][None,:,None,None]).sum(1)  # [B,H,W]
    b = (zT * ab_bins[:,1][None,:,None,None]).sum(1)  # [B,H,W]
    ab_hat = torch.stack([a, b], dim=1)        # [B,2,H,W]
    return ab_hat/128

