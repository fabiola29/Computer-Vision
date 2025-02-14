import os
import time
from glob import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from data import DriveDataset
from model import build_unet
from loss import DiceBCELoss
from utils import seeding, create_dir, epoch_time

def jaccard_index(y_true, y_pred, smooth=1e-10):
    """ Calcola il Jaccard Index (Intersection over Union) """
    y_true = (y_true > 0.5).float()
    y_pred = (y_pred > 0.5).float()

    intersection = (y_true * y_pred).sum(dim=(1, 2, 3))
    union = y_true.sum(dim=(1, 2, 3)) + y_pred.sum(dim=(1, 2, 3)) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return jaccard.mean().item()

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    epoch_jaccard = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_jaccard += jaccard_index(y, y_pred)

    epoch_loss = epoch_loss / len(loader)
    epoch_jaccard = epoch_jaccard / len(loader)
    return epoch_loss, epoch_jaccard

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    epoch_jaccard = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            epoch_jaccard += jaccard_index(y, y_pred)

    epoch_loss = epoch_loss / len(loader)
    epoch_jaccard = epoch_jaccard / len(loader)
    return epoch_loss, epoch_jaccard

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("../new_data/train/image/*"))[:20]
    train_y = sorted(glob("../new_data/train/mask/*"))[:20]

    valid_x = sorted(glob("../new_data/test/image/*"))
    valid_y = sorted(glob("../new_data/test/mask/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 1
    num_epochs = 100
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    loss_fn = DiceBCELoss()

    # Liste per memorizzare i valori
    train_losses = []
    valid_losses = []
    train_jaccards = []
    valid_jaccards = []

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_jaccard = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_jaccard = evaluate(model, valid_loader, loss_fn, device)

        # Memorizzazione dei valori
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_jaccards.append(train_jaccard)
        valid_jaccards.append(valid_jaccard)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f} | Train Jaccard: {train_jaccard:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f} |  Val. Jaccard: {valid_jaccard:.3f}\n'
        print(data_str)

    # Grafico per Loss e Jaccard
    plt.figure(figsize=(12, 6))

    # Grafico della Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, num_epochs + 1), valid_losses, label='Valid Loss', color='red')
    plt.title('Loss durante il Training')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.grid(True) 
    plt.legend()

    # Grafico del Jaccard Index
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_jaccards, label='Train Jaccard', color='blue')
    plt.plot(range(1, num_epochs + 1), valid_jaccards, label='Valid Jaccard', color='red')
    plt.title('Jaccard Index durante il Training')
    plt.xlabel('Epoche')
    plt.ylabel('Jaccard Index')
    plt.grid(True) 
    plt.legend()

    # Mostra il grafico
    plt.tight_layout()
    plt.show()
