import os
import torch
import logging
import cv2
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import albumentations as A

from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset
from dataset import Datasets_2Step
from res_att_unet import ResAttUnet
from utils import EarlyStopping, plot_graph_1
from supcon_cl_loss import BlockConLoss

# Hyperparameter
PRE_LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
PRE_NUM_EPOCHS = 100

# For training in the second step
LABELED_IMG_DIR = "./data/imgs/x_train_1.npz"    # 10 of training images to treat as labeled
LABELED_GTS_DIR = "./data/gts/y_train_1.npz"   # 10 of training images to treat as labeled
UNLABELED_IMG_DIR = "./data/unlabeled_imgs/x_ul_1.npz"       # 90 of training images to treat as unlabeled
PSEUDO_GTS_DIR = "./data/pseudo_gts/pseudo_label_1.npz"  # 90 of training images to treat as pseudo labels

# Checkpoint paths
current_date = str(datetime.now().strftime("%Y_%m_%d"))
checkpoint_name = 'checkpoints/2step_pretraining_checkpoint.pth.tar'


# Model for 2Step Pre-training (model pre-training)
class UNET(nn.Module):
    def __init__(self, num_classes=128):
        super(UNET, self).__init__()
        self.backbone = ResAttUnet(in_channels=3)
        self.prejector = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1),
                                             nn.Conv2d(256, num_classes, kernel_size=1))

    def forward(self, x):
        y = self.backbone(x)
        y = self.prejector(y)
        return y


def main():
    # Data preprocessing for 2Step pre-training
    labeled_imgs = np.load(LABELED_IMG_DIR, allow_pickle=True)['data']
    labeled_gts = np.load(LABELED_GTS_DIR, allow_pickle=True)['data']
    unlabeled_imgs = np.load(UNLABELED_IMG_DIR, allow_pickle=True)['data']
    pseudo_gts = np.load(PSEUDO_GTS_DIR, allow_pickle=True)['data']
    train_imgs_2step = np.concatenate([labeled_imgs, unlabeled_imgs], axis=0)
    train_gts_2step = np.concatenate([labeled_gts, pseudo_gts], axis=0)
    train_data_2step = Datasets_2Step(train_imgs_2step, train_gts_2step, transform=True)

    # Dataloader
    train_loader = DataLoader(train_data_2step, batch_size=BATCH_SIZE, shuffle=True)

    # Model for training
    model = UNET(num_classes=128).to(DEVICE)
    loss_fn = BlockConLoss(temperature=0.1, block_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=PRE_LEARNING_RATE, weight_decay=1e-5)

    # Load Checkpoint
    checkpoint = 'checkpoints/1step_pretraining_checkpoint.pth.tar'
    stats = torch.load(checkpoint)
    model_state = stats['model_state_dict']
    model.backbone.encoder.load_state_dict(model_state)

    logging.disable(logging.NOTSET)
    logging.basicConfig(filename=os.path.join('log_file', 'pre_training_2step.log'), level=logging.DEBUG)
    logging.info(f"\n\nStart SupCon training for {PRE_NUM_EPOCHS} epochs {current_date}")
    early_stopping = EarlyStopping(patience=100, verbose=True, indicator='loss')

    losses = []
    for epoch in range(PRE_NUM_EPOCHS):
        epoch_start_time = time.time()  # Start time for the epoch
        model.train()
        training_loss = 0
        print(f"Epoch: {(epoch + 1)} / {PRE_NUM_EPOCHS}")
        loop = tqdm(dataloader)
        for i, (img_1, img_2, gt_1, gt_2) in enumerate(loop):
            img_1, img_2 = img_1.float(), img_2.float()
            gt_1, gt_2 = gt_1.long(), gt_2.long()

            imgs = torch.cat([img_1, img_2], dim=0)
            labels = torch.cat([gt_1, gt_2], dim=0).squeeze(1)

            with torch.cuda.amp.autocast():
                imgs = imgs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            bsz = imgs.shape[0] // 2
            features = model(imgs)
            features = F.normalize(features, p=2, dim=1)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            l1, l2 = torch.split(labels, [bsz, bsz], dim=0)
            labels = torch.cat([l1.unsqueeze(1), l2.unsqueeze(1)], dim=1)
            loss = loss_fn(features, labels)

            if loss.mean() == 0:
                continue
            mask = (loss != 0)
            mask = mask.int().cuda()
            loss = (loss * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())
            training_loss += loss.item()

        epoch_loss = training_loss / len(dataloader)

        # End time for the epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time  # Time taken for the epoch
        epoch_duration_formatted = format_time(epoch_duration)  # Format the epoch duration into HH:MM:SS

        if torch.isnan(loss).any():
            print("Loss in NaN, training stoped")
            break

        losses.append(epoch_loss)
        logging.debug(f"Epoch: {epoch + 1}\t\tLoss: {epoch_loss}")
        print(f"Epoch: {epoch + 1}\tLoss: {epoch_loss:.4f}\tEpoch time: {epoch_duration_formatted}")

        checkpoint = {
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'model_state_dict': model.backbone.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        early_stopping(epoch_loss, model, checkpoint, checkpoint_name)
        if early_stopping.early_stop:
            print("Early stopped")
            logging.disable(logging.CRITICAL)
            break

        if epoch == PRE_NUM_EPOCHS - 1:
            # torch.save(checkpoint, checkpoint_name)
            logging.disable(logging.CRITICAL)
            break


if __name__ == '__main__':
    main()
