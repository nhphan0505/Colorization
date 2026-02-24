import torch
import torch.nn as nn
def GAN_trainer(epoch, G, D, dataloader, optimizer_G, optimizer_D, criterion_G, criterion_D, device):
    total_loss_G, total_adv_G, total_rec_G = 0.0, 0.0, 0.0
    total_loss_D = 0.0
    
    for L, ab in dataloader:
        L, ab = L.to(device), ab.to(device)

        # Train Discriminator
        for _ in range(1):
            optimizer_D.zero_grad()
            ab_fake = G(L).detach()
            d_real = D(torch.cat([L, ab], dim=1))
            d_fake = D(torch.cat([L, ab_fake], dim=1))
            loss_D = criterion_D(d_real, d_fake)

            loss_D.backward()
            optimizer_D.step()
            total_loss_D += loss_D.item() * L.size(0)

        # Train Generator
        for _ in range(2 + epoch // 300):
            optimizer_G.zero_grad()
            ab_fake = G(L)
            d_fake = D(torch.cat([L, ab_fake], dim=1))
            loss_G, adv_G, rec_G = criterion_G(d_fake, ab_fake, ab)

            loss_G.backward()
            optimizer_G.step()
            total_loss_G += loss_G.item() * L.size(0) / (2 + epoch // 300)
            total_adv_G += adv_G.item() * L.size(0) / (2 + epoch // 300)
            total_rec_G += rec_G.item() * L.size(0) / (2 + epoch // 300)
    return total_loss_G / len(dataloader.dataset), total_adv_G / len(dataloader.dataset), total_rec_G / len(dataloader.dataset), total_loss_D / len(dataloader.dataset)