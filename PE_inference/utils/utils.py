import numpy as np
from PIL import Image

def compose_im(im_arr, alpha, shape=256):
    for i in range(shape):
        for j in range(shape):
            im_arr[i, j, :] = (1-alpha[i, j])*im_arr[i, j, :] + alpha[i, j]*np.array([0., 255., 0.])
    im_cmp = Image.fromarray(im_arr.astype('uint8')).resize((shape, shape))
    return im_cmp

def nms(dets, probs, embs=None, top_n=100, lim=0.0):

    if embs is None:
        dets = np.concatenate([dets, probs[:, np.newaxis]], axis=-1)
    else:
        dets = np.concatenate([dets, probs[:, np.newaxis], embs], axis=-1)
    dets = np.array(sorted(dets, key=lambda x: x[2], reverse=True))[:top_n, :]
    for i in range(dets.shape[0]):
        for j in range(i, dets.shape[0]):
            if (dets[i, 2] != -1) and (dets[j, 2] != -1) and (i != j):
                dist = np.linalg.norm(dets[i, :2]-dets[j, :2])
                if dist <= lim:
                    dets[j, 2] = -1

    to_keep = (dets[:, 2] != -1)
    new_probs = dets[to_keep, 2]
    new_dets = dets[to_keep, :2]
    if embs is not None:
        embs = dets[to_keep, 3:]
    return new_dets, new_probs, embs

    