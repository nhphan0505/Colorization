import numpy as np
import numpy as np
from skimage.color import lab2rgb

def lab_to_rgb(L, ab):
    """
    Decode LAB tensors to RGB image for visualization.

    Args:
        L : (1, H, W) tensor, range [0,1]
        ab: (2, H, W) tensor, range [-1,1]

    Returns:
        rgb image as numpy array (H, W, 3), range [0,1]
    """
    if L.dim() == 4 and ab.dim() == 4:
        L = L.permute(0, 2, 3, 1).cpu().detach().numpy()
        ab = ab.permute(0, 2, 3, 1).cpu().detach().numpy()

        # scale back
        L = L * 100.0
        ab = ab * 128.0

        lab = np.concatenate([L, ab], axis=-1)   # (H,W,3)

        rgb = np.array([lab2rgb(lab[i]) for i in range(lab.shape[0])])
        return np.clip(rgb, 0.0, 1.0)
    
    if L.dim() == 3 and ab.dim() == 3:
        L = L.permute(1, 2, 0).cpu().detach().numpy()
        ab = ab.permute(1, 2, 0).cpu().detach().numpy()

        # scale back
        L = L * 100.0
        ab = ab * 128.0

        lab = np.concatenate([L, ab], axis=-1)   # (H,W,3)

        rgb = lab2rgb(lab)
        return np.clip(rgb, 0.0, 1.0)
    
    return None