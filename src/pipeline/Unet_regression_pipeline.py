import yaml
import torch
import torch.nn as nn
from src.data import *
from src.utils import *
from src.models import Unet_Regression
from src.train import trainer

if __name__ == "__main__":
    set_seed(42)
    with open('configs/unet_regression.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_dataset = Cifar10LAB(train=True, transform=None)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config['batch_size'], 
                                  shuffle=True,
                                  num_workers=6,
                                  pin_memory=True,
                                  persistent_workers=True)
    print("1. Dataset and Dataloader are ready")

    model = Unet_Regression().to(config['device'])
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    print("2. Model, Loss Function and Optimizer are ready")

    print("3. Starting Training...")
    model.train()
    for epoch in range(config['num_epochs']):
        avg_loss = trainer(model, train_dataloader, optimizer, criterion, config['device'])
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"{config['model_save_path']}/unet_regression_epoch_{epoch+1}.pth")
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {avg_loss:.6f}")
    print("Training Completed.")
