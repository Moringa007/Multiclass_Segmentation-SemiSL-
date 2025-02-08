import os
import numpy as np
import torch
import logging
import cv2
import torch.nn as nn
from tqdm import tqdm
import albumentations as A
import segmentation_models_pytorch as smp

from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset, Dataset
from dataset import Datasets_3Step
from res_att_unet import ResAttUnet
from utils import EarlyStopping, plot_graph, plot_graph_2
from validate_function import validate_model, dice_loss
from dice_loss import SoftDiceLoss

# Hyperparameter
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 150
SEED = 42
NUM_CLASSES = 8

# For training in the second step
LABELED_IMG_DIR = "./data/imgs/x_train_1.npz"    # 10 of training images to treat as labeled
LABELED_GTS_DIR = "./data/gts/y_train_1.npz"   # 10 of training images to treat as labeled
VAL_IMG_DIR = "./data/val_imgs.npz"    # 20% of whole dataset for validation
VAL_GTS_DIR = "./data/val_gts.npz"   # 20% of whole dataset for validation

# Checkpoint paths
current_date = str(datetime.now().strftime("%Y_%m_%d"))
checkpoint_name = 'checkpoints/3step_training_checkpoint_.pth.tar'

# Model for downstream training
class FULL_UNET(nn.Module):
    def __init__(self, num_class=5):
        super(FULL_UNET, self).__init__()
        self.backbone = ResAttUnet(in_channels=3)
        self.projection_head = nn.Conv2d(64, num_class, kernel_size=1)

    def forward(self, x):
        y = self.backbone(x)
        output = self.projection_head(y)

        return output



def main():
    # Load dataset
    train_imgs = np.load(LABELED_IMG_DIR, allow_pickle=True)['data']
    train_gts = np.load(LABELED_GTS_DIR, allow_pickle=True)['data']
    val_imgs = np.load(VAL_IMG_DIR, allow_pickle=True)['data']
    val_gts = np.load(VAL_GTS_DIR, allow_pickle=True)['data']
    train_dataset = Datasets_3Step(train_imgs, train_gts, transform=True)
    val_dataset = Datasets_3Step(val_imgs, val_gts, transform=False)

    # Dataloader for training
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # Model for training
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model = FULL_UNET(num_class=5).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    ce_loss = nn.CrossEntropyLoss()
    dc_loss = SoftDiceLoss(batch_dice=True, do_bg=False, smooth=1.0, apply_nonlin=torch.nn.Softmax(dim=1))
    scaler = torch.cuda.amp.GradScaler()

    # Load Checkpoint
    if pre_trained:
        checkpoint = 'checkpoints/2step_pretraining_checkpoint.pth.tar'
        stats = torch.load(checkpoint)
        model_state = stats['model_state_dict']
        model.backbone.load_state_dict(model_state)
        "Segmenation Training"
    else:
        "Baseline Training"

    logging.disable(logging.NOTSET)
    logging.basicConfig(filename=os.path.join('log_file', 'training.log'), level=logging.DEBUG)
    early_stopping = EarlyStopping(patience=30, verbose=True, indicator='dice')

    loss_trend = []
    val_loss_trend = []
    accuracy_trend = []
    dice_trend = []
    iou_trend = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        training_loss = 0
        loop = tqdm(train_loader)
        print(f"============================ Epoch: {epoch + 1}/{NUM_EPOCHS} ============================")
        for idx, (x, y) in enumerate(loop):
            x = x.float().to(DEVICE)
            y = y.long().to(DEVICE)
            y = y.squeeze(1)

            # forward
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = ce_loss(pred, y) + dc_loss(pred, y)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())
            training_loss += loss.item()
        epoch_loss = training_loss / len(train_loader)
        if torch.isnan(loss).any():
            print("Loss is nan, training stopped")
            break

            # checkpoints
        checkpoint = {
            'epoch': epoch + 1,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        # check accuracy
        stats = validate_model(model, val_loader, DEVICE, 5)
        if stats is None:
            print("Validation failed, stats is None")
            break
        acc = stats.get('acc', 0)
        iou = stats.get('iou', 0)
        dice = stats.get('dice', 0)
        val_loss = stats.get('val_loss', 0)
        val_loss_trend.append(val_loss)

        accuracy_trend.append(acc)
        iou_trend.append(iou)
        dice_trend.append(dice)
        loss_trend.append(epoch_loss)
        logging.debug(f"Epoch: {epoch + 1}\tLoss: {epoch_loss}\tIoU: {iou}\tDice: {dice}")
        print(f"train_loss: {epoch_loss:.4f}\tVal_loss: {val_loss:.4f}\tIoU Score: {iou:.2f}\tDice Score: {dice:.2f}")

        early_stopping(dice, model, checkpoint, checkpoint_name)

        if early_stopping.early_stop:
            logging.disable(logging.CRITICAL)
            plot_graph(iou_trend, dice_trend)
            plot_loss(loss_trend, val_loss_trend)
            print("Early stopped")
            break

        if epoch == NUM_EPOCHS - 1:
            logging.disable(logging.CRITICAL)
            plot_graph(iou_trend, dice_trend)
            plot_loss(loss_trend, val_loss_trend)
            break


if __name__ == '__main__':
    main()