import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data import *
from src.utils.seed import set_seed
from src.models import UNetGenerator, PatchDiscriminator
from src.loss import GANLoss
from src.train import GAN_trainer
if __name__ == "__main__":
    set_seed(42)
    with open('configs/gan.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    train_dataset = Cifar10LAB(train=True, transform=train_transform())
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config['batch_size'], 
                                  shuffle=True, 
                                  num_workers=6, 
                                  pin_memory=True,
                                  persistent_workers=True)
    print("1. Dataset and Dataloader are ready")
    
    G = UNetGenerator().to(config['device'])
    D = PatchDiscriminator().to(config['device'])
    optimizer_G = torch.optim.AdamW(G.parameters(), lr=config['G_learning_rate'], betas=(0.5, 0.999))
    optimizer_D = torch.optim.AdamW(D.parameters(), lr=config['D_learning_rate'], betas=(0.5, 0.999))
    criterion_G = GANLoss().to(config['device']).g_loss
    criterion_D = GANLoss().to(config['device']).d_loss
    print("2. Models, Loss Functions and Optimizers are ready")

    print("3. Starting Training...")
    
    def dcgan_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
    G.apply(dcgan_init); D.apply(dcgan_init)

    G.train(); D.train()
    
    for epoch in range(config['num_epochs']):
        avg_loss_G, avg_loss_adv, avg_loss_rec, avg_loss_D = GAN_trainer(epoch, G, D, train_dataloader, optimizer_G, optimizer_D, criterion_G, criterion_D, config['device'])
        if (epoch + 1) % config['log_interval'] == 0:
            torch.save({"G": G.state_dict(),
                       "D": D.state_dict()},
                       os.path.join(config['model_save_path'], f"gan_epoch_{epoch+1}.pth"))
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Generator Loss: {avg_loss_G:.6f}, Adversarial Loss: {avg_loss_adv:.6f}, Reconstruction Loss: {avg_loss_rec:.6f}, Discriminator Loss: {avg_loss_D:.6f}")
    print("Training Completed.")