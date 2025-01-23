import torch
import numpy as np

def confusion_matrix(pred, true):
    confusion_vector = pred / true
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()
    return tn, fp, fn, tp


def calc_scores(pred, true, threshold=0.5, reduction='mean'):
    assert pred.shape == true.shape
    assert pred.ndim == true.ndim == 4
    pred = (pred > threshold).to(float)

    cm = [confusion_matrix(pred[:, i].flatten(), true[:, i].flatten()) for i in range(true.shape[1])]

    accuracy = []
    specificity = []
    sensitivity = []
    precision = []
    f1 = []
    iou = []

    for i in range(len(cm)):
        tn, fp, fn, tp = cm[i]
        accuracy.append((tn + tp) / (tn + fp + fn + tp)) if (tn + fp + fn + tp) != 0 else accuracy.append(0)
        specificity.append(tn / (tn + fp)) if tn + fp != 0 else specificity.append(0)
        sensitivity.append(tp / (tp + fn)) if tp + fn != 0 else sensitivity.append(0)
        precision.append(tp / (tp + fp)) if tp + fp != 0 else precision.append(0)
        f1.append(2 * tp / (2 * tp + fn + fp + 1e-4))
        iou.append(tp / (tp + fp + fn + 1e-4))

    stats = {
                'acc': accuracy,
                'spec': specificity,
                'sens': sensitivity,
                'pre': precision,
                'f1': f1,
                'iou': iou,
            }

    if reduction == 'mean':
        stats = {key: val for key, val in zip(list(stats.keys()), np.array(list(stats.values())).mean(axis=1))}
    elif reduction == 'none':
        stats = {key: val for key, val in zip(list(stats.keys()), np.array(list(stats.values())))}
    return stats
