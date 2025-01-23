import torch
import torchvision
import cv2
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt

from eval_metrics import calc_scores
import albumentations as A
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, indicator='loss'):
        """
        Args:
             patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.indicator = indicator
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf  # When using val loss as indicator
        self.dice_score_max = -np.Inf   # When using dice score as indicator

    def __call__(self, val_loss, model, stat=None, filename=None):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, stat, filename)
        elif score > self.best_score and self.indicator == 'loss':
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        elif score < self.best_score and self.indicator == 'dice':
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, stat, filename)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, stat, filename):

        if self.verbose:
            if self.indicator == 'loss':
                print(f"Validation loss decreased: ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model ...") # loss indicator
                print("=> Saved checkpoint")
                torch.save(stat, filename)
                self.val_loss_min = val_loss
            elif self.indicator == 'dice':
                print(f"Dice score increased: ({self.dice_score_max:.4f} --> {val_loss:.4f}). Saving model ...") # dice indicator
                print("=> Saved checkpoint")
                torch.save(stat, filename)
                self.dice_score_max = val_loss


def remove_module_state_dict(state_dict):
    """ Clean state_dict keys if original state dict was saved from DistribuedDataParallel and
    loaded without """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
        return new_state_dict


# Function for plotting Accuracy, IoU, and Dice Score Trend
def plot_graph(plot_data_2, plot_data_3):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(plot_data_2, label='IoU Score')
    axs[0].set_title("IoU Score Trend")
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('Percentage')
    axs[0].grid()

    axs[1].plot(plot_data_3, label='Dice Score')
    axs[1].set_title("Dice Score Trend")
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('Percentage')
    axs[1].grid()

    plt.tight_layout()
    plt.show()
    fig.savefig('graph.png')


def check_accuracy(loader, model, device="cuda"):
    iou_score = 0
    dice_score = 0
    pixel_accuracy = 0
    
    model.eval()

    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device).to(torch.float32)
            y = y.to(device)

            preds = model(x)
            preds = torch.nn.functional.softmax(preds, dim=1)
            pred_labels = torch.argmax(preds, dim=1)
            pred_labels = pred_labels.unsqueeze(1).float()
            #preds = torch.sigmoid(preds).data > 0.5
            stat = calc_scores(pred_labels, y)

            accuracy = stat["acc"]
            iou = stat['iou']
            iou_score += iou
            dice = stat['f1']
            dice_score += dice

            accuracies = accuracy * 100
            ious = iou_score / len(loader) * 100
            dices = dice_score / len(loader) * 100
        #print(f"Accuracy : {accuracy * 100}, IoU : {iou_score / len(loader) * 100}, Dice score: {dice_score / len(loader) * 100}")

        model.train()

        return accuracies, ious, dices


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds >= 0.5).float()

        preds_rgb = preds.repeat(1, 3, 1, 1)
        y_rgb = y.repeat(1, 3, 1, 1)
        torchvision.utils.save_image(
            preds_rgb * 255, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y_rgb * 255, f"{folder}{idx}.png")

    model.train()


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def plot_loss(data_1, data_2):
    plt.figure(figsize=(10, 6))

    plt.plot(data_1, label='Training Loss', linestyle='-', color='blue')
    plt.plot(data_2, label='Validation Loss', linestyle='--', color='green')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Over Iterations')
    plt.legend()
    plt.grid()

    plt.show()


# Helper function to convert seconds to HH:MM:SS
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


selected_class_indice = [0, 1, 2, 3, 4, 5, 6, 7]
selected_class_rgb = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]

def color_code_segment(image):
    color_code = np.array(selected_class_rgb)
    x = color_code[image.astype(int)]
    return x


def create_overlay(image, mask, alpha=0.7):
    # Ensure the image and mask are in the same shape
    if mask.ndim == 2:  # If mask is grayscale, convert to RGB
        mask = np.stack([mask] * 3, axis=-1)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Blend the images
    overlay = alpha * mask + (1 - alpha) * image
    return np.clip(overlay, 0, 1)  # Ensure values stay in [0, 1]

import random

def visualize(images, masks):
    fontsize = 18
    num_images = min(5, len(images))  # Limit to 10 images if there are more

    # Randomly select indices from the whole dataset
    indices = random.sample(range(len(images)), num_images)

    f, ax = plt.subplots(3, num_images, figsize=(num_images * 4, 10))

    for i in range(num_images):
        idx = indices[i]
        image = images[idx] / 255.0 
        ax[0, i].imshow(image.transpose(1, 2, 0), cmap='gray')
        ax[0, i].set_title(f'Image Index: {idx}', fontsize=fontsize)
        ax[0, i].axis('off')

        ax[1, i].imshow(color_code_segment(masks[idx]))
        ax[1, i].set_title(f'Ground Truth Index: {idx}', fontsize=fontsize)
        ax[1, i].axis('off')

        overlay = create_overlay(image.transpose(1, 2, 0), color_code_segment(masks[idx]))
        ax[2, i].imshow(overlay)
        ax[2, i].set_title(f"Overlay Index: {idx}", fontsize=fontsize)
        ax[2, i].axis('off')

    plt.tight_layout()
    plt.show()


