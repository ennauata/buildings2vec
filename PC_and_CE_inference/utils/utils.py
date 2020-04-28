from numpy.random import randn
#import ref
import torch
import numpy as np

def nms(junctions, junc_confs, thetas, theta_confs, nms_thresh=4.0):

    inds = np.argsort(junc_confs)[::-1]
    junc_confs_sorted = np.array(junc_confs[inds])
    juncs_sorted = np.array(junctions[inds])
    thetas_sorted = np.array(thetas[inds])
    theta_confs_sorted = np.array(theta_confs[inds])

    # get loop 
    dists = np.apply_along_axis(np.linalg.norm, 2,\
        juncs_sorted[:, None, :] - juncs_sorted[None, :, :])

    # apply nms
    keep_track = np.zeros(junc_confs_sorted.shape[0])
    nms_inds = []
    for i in range(junc_confs_sorted.shape[0]):
        if (keep_track[i] == 0):
            nms_inds.append(i)
            for j in range(junc_confs_sorted.shape[0]):
                if dists[i, j] < nms_thresh:
                    keep_track[j] = 1

    return juncs_sorted[nms_inds], junc_confs_sorted[nms_inds], thetas_sorted[nms_inds], theta_confs_sorted[nms_inds]

def adjust_learning_rate(optimizer, epoch, LR, LR_param):
    #lr = LR * (0.1 ** (epoch // dropLR))
    LR_policy = LR_param.get('lr_policy', 'step')
    if LR_policy == 'step':
        steppoints = LR_param.get('steppoints', [4, 7, 9, 10])
        lrs = LR_param.get('lrs', [0.001, 0.001, 0.001, 0.001, 0.001])
        assert len(lrs) == len(steppoints) + 1
        
        lr = None
        for idx, steppoint in enumerate(steppoints):
            if epoch > steppoint:
                continue
            elif epoch <= steppoint:
                lr = lrs[idx]
                break
        if lr is None:
            lr = lrs[-1]

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
             
def Rnd(x):
  return max(-2 * x, min(2 * x, randn() * x))
  
def Flip(img):
  return img[:, :, ::-1].copy()  
  
def ShuffleLR(x):
  for e in ref.shuffleRef:
    x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
  return x
