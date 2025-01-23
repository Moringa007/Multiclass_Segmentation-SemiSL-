from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss with multi-class balancing for segmentation, excluding background."""

    def __init__(self, temperature=0.7, contrast_mode='all', base_temperature=0.7, background_class=0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.background_class = background_class  # Class to exclude (e.g., background)

    def forward(self, features, labels=None):
        # input features shape: [batch_size, num_views, channels, height, width]
        # input labels shape: [batch_size, num_views, height, width]
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [batch_size, num_views, ...],'
                             'at least 3 dimensions are required')


        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # of size (bsz*v, c, h, w)

        kernels = contrast_feature.permute(0, 2, 3, 1)
        kernels = kernels.reshape(-1, contrast_feature.shape[1], 1, 1)
        logits = torch.div(F.conv2d(contrast_feature, kernels), self.temperature)
        logits = logits.permute(1, 0, 2, 3)
        logits = logits.reshape(logits.shape[0], -1)

        labels = torch.cat(torch.unbind(labels, dim=1), dim=0)
        labels = labels.contiguous().view(-1, 1)
        labels = torch.eq(labels, labels.T).float().to(device)

        # Initialize variables for balanced multi-class loss
        total_loss = 0
        num_classes = int(labels.max().item()) + 1  # Assuming labels are in range [0, num_classes-1]
        valid_class_count = 0  # To track valid classes excluding the background

        # Compute contrastive loss for each class separately, excluding background
        for cls in range(num_classes):
            if cls == self.background_class:
                continue  # Skip background class

            class_mask = (labels == cls).float().to(device)  # Mask for current class
            if class_mask.sum() == 0:
                continue  # Skip if no pixels of this class

            # Mask self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(labels),
                1,
                torch.arange(labels.shape[0]).view(-1, 1).to(device),
                0
            )
            mask = class_mask * logits_mask #* upper_triangular_mask

            # Compute log probabilities for contrastive loss
            exp_logits = torch.exp(logits) * logits_mask #* upper_triangular_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # Calculate mean log-likelihood for positive pairs
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            mean_log_prob_pos = mean_log_prob_pos[mask.sum(1) > 0] # filter valid entries

            # Calculate contrastive loss for the current class
            class_loss = -self.temperature * mean_log_prob_pos.mean()

            total_loss += class_loss  # Accumulate loss for each class
            valid_class_count += 1

        # Average loss across all valid (non-background) classes
        return total_loss / valid_class_count if valid_class_count > 0 else torch.tensor(0.0, device=device)


class BlockConLoss(nn.Module):
    def __init__(self, temperature=0.7, block_size=32):
        super(BlockConLoss, self).__init__()
        self.block_size = block_size
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=temperature)

    def forward(self, features, labels=None):
        # input features: [bsz, num_view, c, h ,w], h & w are the image size
        shape = features.shape
        img_size = shape[-1]
        div_num = img_size // self.block_size
        #if labels.max() == 0:
        #    labels = None

        if labels is not None:
            loss = []
            for i in range(div_num):
                # print("Iteration index:", idx, "Batch_size:", b)
                for j in range(div_num):
                    #print(f"i: {i}, j: {j}")
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    block_features = features[:, :, :, i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]
                    block_labels = labels[:, :, i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]

                    if block_labels.sum() == 0:
                        continue
                    tmp_loss = self.supconloss(block_features, block_labels)
                    loss.append(tmp_loss)

            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            return loss

        else:
            loss = []
            for i in range(div_num):
                # print("Iteration index:", idx, "Batch_size:", b)
                for j in range(div_num):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    block_features = features[:, :, :, i * self.block_size:(i + 1) * self.block_size,
                                     j * self.block_size:(j + 1) * self.block_size]

                    tmp_loss = self.supconloss(block_features)

                    loss.append(tmp_loss)

            loss = torch.stack(loss).mean()
            return loss