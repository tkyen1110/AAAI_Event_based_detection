import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gpu_device=0):
        super(FocalLoss, self).__init__()
        self.gpu_device = gpu_device

    def forward(self, classifications, regressions, anchors, annotations):
        # classifications.shape = [2, (64*64 + 32*32 + 16*16 + 8*8 + 4*4) * num_anchors, num_classes] = [2, 81840, 7]
        # regressions.shape     = [2, (64*64 + 32*32 + 16*16 + 8*8 + 4*4) * num_anchors, 4] = [2, 81840, 4] [x_min, y_min, x_max, y_max]
        # anchors.shape         = [1, (64*64 + 32*32 + 16*16 + 8*8 + 4*4) * num_anchors, 4] = [1, 81840, 4] [x_min, y_min, x_max, y_max]
        # annotations.shape     = [2, 60, 5] [x_min, y_min, x_max, y_max, class_id]

        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :] # shape = [81840, 4] [x_min, y_min, x_max, y_max]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :] # shape = [81840, 7]
            regression = regressions[j, :, :] # shape = [81840, 4]

            bbox_annotation = annotations[j, :, :] # shape = [60, 5]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1] # shape = [num_annotations, 5]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).to(self.gpu_device) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().to(self.gpu_device))

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations
            # anchors[0, :, :].shape       = [num_anchors=81840, 4]
            # bbox_annotation[:, :4].shape = [num_annotations  , 4]
            # IoU.shape = [num_anchors=81840, num_annotations]
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1 # shape = [81840, 7]

            if torch.cuda.is_available():
                targets = targets.to(self.gpu_device)

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :] # shape = [num_anchors=81840, 5]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).to(self.gpu_device) * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(self.gpu_device))
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(self.gpu_device)
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])
                # smooth l1 loss
                regression_loss = torch.where(torch.le(regression_diff, 1.0 / 9.0),
                                              0.5 * 9.0 * torch.pow(regression_diff, 2),
                                              regression_diff - 0.5 / 9.0)
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().to(self.gpu_device))
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)


def calc_iou(a, b):
    # a: anchors          [num_anchors    , 4] = [81840, 4]
    # b: bbox_annotations [num_annotations, 4]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua # [num_anchors, num_annotations] = [81840, num_annotations]

    return IoU
