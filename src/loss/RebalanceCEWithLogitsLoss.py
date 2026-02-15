import torch
import torch.nn.functional as F
from src.color.quantize import soft_encode

class RebalanceCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, class_weights, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer("w", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, target):
        target = soft_encode(target)  # (B, Q, H, W)
        B, Q, H, W = logits.shape
        log_probs = F.log_softmax(logits, dim=1)  # (B,Q,H,W)
        q_star = target.argmax(dim=1)  # (B,H,W)
        ce_per_pixel = -(target * log_probs).sum(dim=1)  # (B,H,W)
        weights = self.w[q_star]  # (B,H,W)
        loss_per_pixel = weights * ce_per_pixel  # (B,H,W)

        return loss_per_pixel.sum() / (weights.sum() + self.eps)
    
