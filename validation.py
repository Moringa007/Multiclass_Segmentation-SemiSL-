import torch
import torch.nn as nn
import torch.nn.functional as F
from dice_loss import SoftDiceLoss
from tqdm import tqdm


def calculate_dice_score(output, target, class_id):
    # Apply softmax and get the predicted class
    pred_mask = F.softmax(output, dim=1)
    pred_mask = torch.argmax(pred_mask, dim=1)

    # Create binary masks for the current class
    pred_mask = (pred_mask == class_id)
    true_mask = (target == class_id)

    # Calculate TP, FP, and FN
    tp = torch.sum(pred_mask & true_mask).float()
    fp = torch.sum(pred_mask & ~true_mask).float()
    fn = torch.sum(~pred_mask & true_mask).float()

    # Calculate Dice Score
    dice_score = (2 * tp) / (2 * tp + fp + fn + 1e-4)

    # Calculate IoU
    iou = tp / (tp + fp + fn + 1e-4)

    # Calculate Accuracy (only for the relevant class)
    acc = tp / (tp + fp + fn + 1e-4)

    return dice_score.item(), iou.item(), acc.item()

def validate_model(model, dataloader, device, num_classes):
    model.eval()
    validating_loss = 0
    total_dice_scores = {class_id: 0.0 for class_id in range(1, num_classes)}
    total_iou_scores = {class_id: 0.0 for class_id in range(1, num_classes)}
    total_acc_scores = {class_id: 0.0 for class_id in range(1, num_classes)}

    ce_loss = nn.CrossEntropyLoss()
    dc_loss = SoftDiceLoss(batch_dice=True, do_bg=False, smooth=1.0, apply_nonlin=torch.nn.Softmax(dim=1))

    with torch.no_grad():
        loop = tqdm(dataloader)
        for inputs, targets in loop:
            inputs = inputs.float().to(device)
            targets = targets.long().to(device)
            targets = targets.squeeze(1)

            # Forward pass
            preds = model(inputs)
            #pred_softmax = F.softmax(preds, dim=1)
            val_loss = ce_loss(preds, targets) + dc_loss(preds, targets)
            validating_loss += val_loss.item()
            loop.set_postfix(loss=val_loss.item())

            # Calculate metrics for each class (excluding background)
            for class_id in range(1, num_classes):
                dice_score, iou, acc = calculate_dice_score(preds, targets, class_id)
                total_dice_scores[class_id] += dice_score
                total_iou_scores[class_id] += iou
                total_acc_scores[class_id] += acc
        loss = validating_loss / len(dataloader)

    # Calculate average metrics
    num_batches = len(dataloader)
    average_dice_scores = {class_id: total_dice_scores[class_id] / num_batches for class_id in range(1, num_classes)}
    average_iou_scores = {class_id: total_iou_scores[class_id] / num_batches for class_id in range(1, num_classes)}
    average_acc_scores = {class_id: total_acc_scores[class_id] / num_batches for class_id in range(1, num_classes)}

    # Calculate mean metrics across classes (excluding background)
    mean_average_dice = sum(average_dice_scores.values()) / len(average_dice_scores) * 100
    mean_average_iou = sum(average_iou_scores.values()) / len(average_iou_scores) * 100
    mean_average_acc = sum(average_acc_scores.values()) / len(average_acc_scores) * 100

    # Store stats for further usage
    stats = {
        'val_loss': loss,
        'acc': mean_average_acc,
        'iou': mean_average_iou,
        'dice': mean_average_dice
    }
    return stats
