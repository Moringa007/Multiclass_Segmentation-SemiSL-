import os
import torch
import logging
import cv2
import torch.nn as nn
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader, Subset, Dataset
from dataset import  Datasets_1Step
from unet_model import MLP
from nt_xent import NTXentLoss
from res_att_unet import ResAttEncoder
from utils import EarlyStopping, plot_graph_1

# Hyperparameter
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 500
SEED = 42
TRAIN_IMG_DIR = "./data/train_imgs.npz"

current_date = str(datetime.now().strftime("%Y_%m_%d"))
checkpoint_name = 'checkpoints/1step_pretraining_checkpoint.pth.tar'


# Model for 1Step pre-training (encoder pre-training)
class Encoder(nn.Module):
    def __init__(self, num_classes=128):
        super(Encoder, self).__init__()
        self.backbone = ResAttEncoder(in_channels=3)
        self.projection_head = MLP(512, num_class=num_classes)
        self.reconstruction_head = nn.ConvTranspose2d(in_channels=512, out_channels=3, kernel_size=8, stride=8)

    def forward(self, x1, x2):
        z_1, _ = self.backbone(x1)
        z_2, _ = self.backbone(x2)
        z_recon_1 = self.reconstruction_head(z_1)
        #z_recon_2 = self.reconstruction_head(z_2)

        z_1 = self.projection_head(z_1)
        z_2 = self.projection_head(z_2)

        return x1, z_recon_1, z_1, z_2


def plot_graph(data_1, data_2, data_3):
    plt.figure(figsize=(10, 5))

    plt.plot(data_1, label='Total Loss', linestyle='-', color='blue')
    plt.plot(data_2, label='Loss_1', linestyle='--', color='green')
    plt.plot(data_3, label='Loss_2', linestyle='-.', color='red')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()
    plt.grid()

    plt.show()


def main():
    # Data loading and pre-processing for 1st step pretraining
    train_imgs_1step = np.load(PRE_TRAIN_IMG_DIR, allow_pickle=True)['data']
    train_data_1step = Datasets_1Step(train_imgs_1step)

    # Dataloader for training
    train_loader = DataLoader(train_data_1step, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model for training
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model = Encoder(num_classes=128).to(DEVICE)
    loss_nt = NTXentLoss(DEVICE, temperature=0.1, use_cosine_similarity=True)
    loss_mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=PRE_LEARNING_RATE, momentum=0.9)

    logging.disable(logging.NOTSET)
    logging.basicConfig(filename=os.path.join('log_file', 'pre_training.log'), level=logging.DEBUG)
    early_stopping = EarlyStopping(patience=25, verbose=True, indicator='loss')

    loss_trend = []
    loss_1_trend = []
    loss_2_trend = []
    beta = 0.1

    for epoch in range(NUM_EPOCHS_1Step):
        model.train()
        epoch_loss = 0
        epoch_loss_1 = 0
        epoch_loss_2 = 0
        training_loss = 0
        running_loss_1 = 0
        running_loss_2 = 0
        loop = tqdm(train_loader)
        for i, (x1, x2) in enumerate(loop):
            x1, x2 = x1.float().to(DEVICE), x2.float().to(DEVICE)

            with torch.cuda.amp.autocast():
                z_real, z_recon, z_anchors, z_positive = model(x1, x2)
                loss_1 = loss_nt(z_anchors, z_positive)
                loss_2 = loss_mse(z_real, z_recon)
                loss = loss_1 * beta + loss_2 * (1 - beta)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())
            training_loss += loss.item()
            epoch_loss = training_loss / len(train_loader)
            running_loss_1 += loss_1.item()
            epoch_loss_1 = running_loss_1 / len(train_loader)
            running_loss_2 += loss_2.item()
            epoch_loss_2 = running_loss_2 / len(train_loader)

        if torch.isnan(loss).any():
            print("Loss is NaN. Stopping training.")
            break

        loss_trend.append(epoch_loss)
        loss_1_trend.append(epoch_loss_1)
        loss_2_trend.append(epoch_loss_2)

        logging.debug(f"Epoch: {epoch + 1}\tLoss: {epoch_loss :.8f}")
        print(f"Epoch: {epoch + 1}\tLOSS: {epoch_loss :.4f}\tLOSS_1: {epoch_loss_1 :.4f}\tLOSS_2: {epoch_loss_2 :.4f}")

        checkpoint = {
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'model_state_dict': model.backbone.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        early_stopping(epoch_loss, model.backbone, checkpoint, checkpoint_name)

        if early_stopping.early_stop:
            print("Early stopped")
            logging.info("Training has finished")
            logging.disable(logging.CRITICAL)
            plot_graph(loss_trend, loss_1_trend, loss_2_trend)
            break

        if epoch == (NUM_EPOCHS_1Step - 1):
            logging.info("Training has finished")
            logging.disable(logging.CRITICAL)
            break

if __name__ == "__main__":
    main()