import torch
import torch.nn.functional as F
from torch import nn
import torchbiomed.loss as bioloss

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        # probs = F.sigmoid(logits)
        probs = logits
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        # target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target - (pred * target)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss


class clDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(clDiceLoss, self).__init__()
        self.smooth = smooth

    def soft_cldice_loss(self, pred, target, target_skeleton=None):
        '''
        inputs shape  (batch, channel, height, width).
        calculate clDice loss
        Because pred and target at moment of loss calculation will be a torch tensors
        it is preferable to calculate target_skeleton on the step of batch forming,
        when it will be in numpy array format by means of opencv
        '''
        cl_pred = self.soft_skeletonize(pred)
        if target_skeleton is None:
            target_skeleton = self.soft_skeletonize(target)
        iflat = self.norm_intersection(cl_pred, target)
        tflat = self.norm_intersection(target_skeleton, pred)
        intersection = (iflat * tflat).sum()
        return 1. - (2. * intersection) / (iflat + tflat).sum()

def CELoss(y, label):
    loss = nn.CrossEntropyLoss()
    return loss(y, label)


def BCELoss(prediction, label):
    masks_probs_flat = prediction.view(-1)
    true_masks_flat = label.float().view(-1)
    loss = nn.BCELoss()(masks_probs_flat, true_masks_flat)
    return loss


def calc_loss(algorithm, prelabel, label):
    dicemetric = SoftDiceLoss()
    jsloss = SoftIoULoss(n_classes=1)
    snakedice = DiceLoss()
    snakecldice = clDiceLoss()
    if algorithm == 'unet' or algorithm == 'deform_unet_v1':
        loss = BCELoss(prelabel, label)
    elif algorithm == 'AttUnet':
        loss = dicemetric(prelabel, label)
    elif algorithm == 'UKAN':
        loss = BCELoss(prelabel, label) + dicemetric(prelabel,label)
    elif algorithm == 'snake':
        loss = snakedice(prelabel, label) * 0.8 + snakecldice(prelabel, label)
    elif algorithm == 'newDU' or algorithm == 'newDU31' or algorithm == 'newDU22':
        loss = BCELoss(prelabel, label) * 0.8 + dicemetric(prelabel,label) + jsloss(prelabel, label) * 0.2

    return loss
