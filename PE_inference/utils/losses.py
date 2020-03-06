import torch
import torch.nn as nn
import torch.nn.functional as F

def balanced_binary_cross_entropy(pred, gt, pos_w=2.0, neg_w=1.0):

    # flatten vectors
    pred = pred.view(-1)
    gt = gt.view(-1)

    # select postive/nevative samples
    pos_ind = gt.nonzero().squeeze(-1)
    neg_ind = (gt == 0).nonzero().squeeze(-1)

    # compute weighted loss
    pos_loss = pos_w*F.binary_cross_entropy(pred[pos_ind], gt[pos_ind], size_average=True)
    neg_loss = neg_w*F.binary_cross_entropy(pred[neg_ind], gt[neg_ind], size_average=True)

    return pos_loss + neg_loss

def mse(coords, coords_gt, prob_gt):

    # flatten vectors
    coords = coords.view(-1, 2)
    coords_gt = coords_gt.view(-1, 2)
    prob_gt = prob_gt.view(-1)

    # select positive samples
    pos_ind = prob_gt.nonzero().squeeze(-1)
    pos_coords = coords[pos_ind, :]
    pos_coords_gt = coords_gt[pos_ind, :]

    return F.mse_loss(pos_coords, pos_coords_gt)