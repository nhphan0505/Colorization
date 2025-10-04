import numpy as np
import torch
import random

def set_seed():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 100% deterministic but slower training
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False