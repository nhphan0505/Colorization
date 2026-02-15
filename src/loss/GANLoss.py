import torch
import torch.nn as nn
class GANLoss(nn.Module):
    """
    Loss = cGAN adversarial + λ * L1.
    - D_loss: log D(L,ab_real) + log(1 - D(L,ab_fake))
    - G_loss: log D(L,ab_fake) + λ * L1(ab_fake, ab_real)
    """
    def __init__(self, lambda_L1=100.0):
        super().__init__()
        self.lambda_L1 = lambda_L1
        self.bce = nn.BCEWithLogitsLoss()
        self.l1  = nn.L1Loss()

    def d_loss(self, d_real, d_fake):
        """
        d_real: D(L, ab_real) -> [B,1,H,W]
        d_fake: D(L, ab_fake.detach()) -> [B,1,H,W]
        """
        loss_real = self.bce(d_real, torch.ones_like(d_real))
        loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
        return loss_real + loss_fake

    def g_loss(self, d_fake, ab_fake, ab_real):
        """
        d_fake: D(L, ab_fake) -> [B,1,H,W]
        """
        adv = self.bce(d_fake, torch.ones_like(d_fake))
        rec = self.l1(ab_fake, ab_real) * self.lambda_L1
        total = adv + rec
        return total, adv.detach(), rec.detach()