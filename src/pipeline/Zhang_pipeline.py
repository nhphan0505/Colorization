import os
import yaml
import torch
import torch.nn as nn
from src.data import *
from src.utils import *
from src.models import DeepCNN
from src.loss import RebalanceCEWithLogitsLoss
from src.color.prior import color_weight
from src.train import train_one_epoch
if __name__ == "__main__":
    set_seed(42)
    with open('configs/zhang.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    train_dataset = Cifar10LAB(train=True, transform=None)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config['batch_size'], 
                                  shuffle=True, 
                                  num_workers=6, 
                                  pin_memory=True,
                                  persistent_workers=True)
    print("1. Dataset and Dataloader are ready")

    model = DeepCNN().to(config['device'])
    if os.path.exists(config['weights_path']):
        weights = torch.load(config['weights_path'], map_location=config['device'])
    else:
        weights = color_weight(train_dataloader)
    criterion = RebalanceCEWithLogitsLoss(class_weights=weights).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    print("2. Model, Loss Function and Optimizer are ready")

    print("3. Starting Training...")
    for epoch in range(config['num_epochs']):
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, config['device'])
        scheduler.step()
        if (epoch + 1) % config['log_interval'] == 0:
            torch.save({"model": model.state_dict(),
                       "scheduler": scheduler.state_dict(),
                       "optimizer": optimizer.state_dict(),},
                       os.path.join(config['model_save_path'], f"zhang_epoch_{epoch+1}.pth"))
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {avg_loss:.6f}")
    print("Training Completed.")