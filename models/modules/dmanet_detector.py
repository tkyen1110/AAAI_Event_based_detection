import torch
import torch.nn as nn
from models.functions.box_utils import BBoxTransform, ClipBoxes
from torchvision.ops import nms


class DMANet_Detector(nn.Module):
    def __init__(self, conf_threshold, iou_threshold, gpu_device):
        super(DMANet_Detector, self).__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.gpu_device = gpu_device
        self.regressBoxes = BBoxTransform(gpu_device=gpu_device)
        self.clipBoxes = ClipBoxes()

    def forward(self, classification, regression, anchors, img_batch):
        # classification.shape = [2, (64*64 + 32*32 + 16*16 + 8*8 + 4*4) * num_anchors, num_classes] = [2, 81840, 7]
        # regression.shape = [2, (64*64 + 32*32 + 16*16 + 8*8 + 4*4) * num_anchors, 4] = [2, 81840, 4]
        # anchors.shape = [1, (64*64 + 32*32 + 16*16 + 8*8 + 4*4) * num_anchors, 4] = [1, 81840, 4]
        # img_batch.shape = [2, D=16, H=512, W=512]
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        finalResult = [[], [], []]

        finalScores = torch.Tensor([])
        finalAnchorBoxesIndexes = torch.Tensor([]).long()
        finalAnchorBoxesCoordinates = torch.Tensor([])

        if torch.cuda.is_available():
            finalScores = finalScores.to(self.gpu_device)
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.to(self.gpu_device)
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.to(self.gpu_device)

        for i in range(classification.shape[2]):
            scores = torch.squeeze(classification[:, :, i])
            scores_over_thresh = (scores > self.conf_threshold)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, self.iou_threshold)

            finalResult[0].extend(scores[anchors_nms_idx])
            finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
            finalResult[2].extend(anchorBoxes[anchors_nms_idx])

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            if torch.cuda.is_available():
                finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.to(self.gpu_device)

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        if len(finalScores):
            finalScores = finalScores.unsqueeze(-1)
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.type(torch.float32).unsqueeze(-1)
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates
            return torch.cat([finalAnchorBoxesCoordinates, finalScores, finalAnchorBoxesIndexes], dim=1)
        else:  # empty
            return torch.tensor([]).reshape(-1, 6)
