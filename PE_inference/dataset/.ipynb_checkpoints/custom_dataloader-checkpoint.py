import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pickle as p
import glob
from random import randint

class ComposedImageData(Dataset):

    def __init__(self, rgb_folder, annot_folder, id_list, mean, std, augment=False, depth_folder=None, gray_folder=None, surf_folder=None):
        self._data_refs = id_list
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.annot_folder = annot_folder
        self.gray_folder = gray_folder
        self.surf_folder = surf_folder
        self.augment = augment
        self.mean = mean
        self.std = std

    def __getitem__(self, index):

        # load image + annots
        im_path = os.path.join(self.rgb_folder, self._data_refs[index]+'.jpg')
        im = Image.open(im_path)

        if self.depth_folder is not None:
            dp_path = os.path.join(self.depth_folder, self._data_refs[index]+'.jpg')
            dp_im = Image.open(dp_path).convert('L')

        # if self.gray_folder is not None:
        #     gr_path = os.path.join(self.gray_folder, self._data_refs[index]+'.jpg')
        #     gr_im = Image.open(gr_path).convert('L')

        # if self.surf_folder is not None:
        #     sf_path = os.path.join(self.surf_folder, self._data_refs[index]+'.jpg')
        #     sf_im = Image.open(sf_path).convert('RGB')
        corners_annot, edges_annot = self.get_annots(index)
        im_ed = self.calc_edge_gt(edges_annot)
        if self.augment:
            rot = randint(0, 359)
            flip = randint(0, 1) == 1

            # rotate and flip image + corners
            im_ed = self.calc_edge_gt(edges_annot, flip=flip, rot=rot)
            im = im.rotate(rot)
            #im_ed = im_ed.rotate(rot)
            if flip:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                #im_ed = im_ed.transpose(Image.FLIP_LEFT_RIGHT)

            # add depth
            if self.depth_folder is not None:
                dp_im = dp_im.rotate(rot)
                if flip:
                    dp_im = dp_im.transpose(Image.FLIP_LEFT_RIGHT)

            # # add gray
            # if self.gray_folder is not None:
            #     gr_im = gr_im.rotate(rot)
            #     if flip:
            #         gr_im = gr_im.transpose(Image.FLIP_LEFT_RIGHT)

            # # add surf
            # if self.surf_folder is not None:
            #     sf_im = sf_im.rotate(rot)
            #     if flip:
            #         sf_im = sf_im.transpose(Image.FLIP_LEFT_RIGHT)

        # convert to numpy array
        im = np.array(im).transpose((2, 0, 1))/255.0
        im = (im-np.array(self.mean)[:, np.newaxis, np.newaxis])/np.array(self.std)[:, np.newaxis, np.newaxis]
        ed = np.array(im_ed)/255.

        if self.depth_folder is not None:
            dp_im = np.array(dp_im)/255.0
            im = np.concatenate([im, dp_im[np.newaxis, :, :]], axis=0)

        # if self.gray_folder is not None:
        #     gr_im = np.array(gr_im)/255.0
        #     im = np.concatenate([im, gr_im[np.newaxis, :, :]], axis=0)

        # if self.surf_folder is not None:
        #     sf_im = np.array(sf_im).transpose((2, 0, 1))/255.0
        #     im = np.concatenate([im, sf_im], axis=0)

        # convert to tensor
        im = torch.from_numpy(im)
        ed = torch.from_numpy(ed)
        return im, ed

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self._data_refs)

    # rotate coords
    def rotate(self, image_shape, xy, angle):
        org_center = (image_shape-1)/2.
        rot_center = (image_shape-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
                -org[0]*np.sin(a) + org[1]*np.cos(a) ])
        new = new+rot_center
        return new

    def get_annots(self, index):

        # load ground-truth
        gt_path = os.path.join(self.annot_folder, self._data_refs[index]+'.npy')
        v_set = np.load(open(gt_path, "rb"),  encoding='bytes', allow_pickle=True)
        v_set = dict(v_set[()])
        v_set = v_set[b'graph']
        return v_set.keys(), v_set

    # def calc_edge_gt(self, annot, shape=256):
    #     im = Image.fromarray(np.zeros((shape, shape)))
    #     for c in annot:
    #         draw = ImageDraw.Draw(im)
    #         for n in annot[c]:
    #             dist = np.sqrt((n[0] - c[0])**2 + (n[1] - c[1])**2)
    #             draw.line((c[0], c[1], n[0], n[1]), fill=dist, width=1)
    #     return im

    def calc_edge_gt(self, annot, flip=False, rot=0, shape=256):

        edge_set = set()
        for v1 in annot:
            for v2 in annot[v1]:
                x1, y1 = v1
                x2, y2 = v2

                x1, y1 = self.flip_and_rotate((x1, y1), flip, rot)
                x2, y2 = self.flip_and_rotate((x2, y2), flip, rot)

                # make an order
                if x1 > x2:
                    x1, x2, y1, y2 = x2, x1, y2, y1  # swap
                elif x1 == x2 and y1 > y2:
                    x1, x2, y1, y2 = x2, x1, y2, y1  # swap
                else:
                    pass
                edge = (x1, y1, x2, y2)
                edge_set.add(edge)

        im = Image.fromarray(np.zeros((shape, shape)))
        draw = ImageDraw.Draw(im)
        for e in edge_set:

            # compute angle
            x1, y1, x2, y2 = e

            # pc = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
            # pp = np.array([0, 1])
            # pr = np.array([x1, y1]) if x1 >= x2 else np.array([x2, y2])
            # pr -= pc
            # cosine_angle = np.dot(pp, pr) / (np.linalg.norm(pp) * np.linalg.norm(pr))
            # angle = np.arccos(cosine_angle)
            # angle = 180.0 - np.degrees(angle)
            # delta_degree = 10.0
            # n_bins = 18
            # angle_bin = int(angle/delta_degree)%n_bins
            # angle_bin += 1
            draw.line((x1, y1, x2, y2), fill='white', width=4)            

        return im

    def flip_and_rotate(self, v, flip, rot, shape=256.):
        v = self.rotate(np.array((shape, shape)), v, rot)
        if flip:
            x, y = v
            v = (shape/2-abs(shape/2-x), y) if x > shape/2 else (shape/2+abs(shape/2-x), y)
        return v




