from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from PIL import Image, ImageDraw
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import random
import glob
import pickle as p

class BuildingsDatasetTest(object):
    def __init__(self, img_dir, ann_file, id_file, transforms=None, split='test'):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.split = split
        self.building_id_file = id_file
        with open(self.building_id_file) as f:
            self.building_ids = [x.strip() for x in f.readlines()]
        self.building_ids = self.building_ids
        self.data_folder = '/home/nelson/Workspace/building_reconstruction/working_model/region_boundary_estimator/data'
        self.region_folder = '/home/nelson/Workspace/cities_dataset/regions_with_bkg'
        self.sample_list = []
        
        # list regions
        for _id in self.building_ids:
            region_path = '{}/{}.npy'.format(self.region_folder, _id)
            regions_det = np.load(region_path)      

            # create region pairs
            if regions_det.shape[0] >= 2:
                for i in range(regions_det.shape[0]):
                    for j in range(regions_det.shape[0]):
                        if i > j:
                            self.sample_list.append([_id, i, j])
        return

    def __getitem__(self, idx):
        
        _building_id, reg_i_idx, reg_j_idx = self.sample_list[idx]

        # load data
        rgb_im = Image.open('{}/{}.jpg'.format(self.img_dir, _building_id))

        # load regions
        region_path = '{}/{}.npy'.format(self.region_folder, _building_id)
        regions_det = np.load(region_path)

        # create region pairs
        reg_i = regions_det[reg_i_idx][np.newaxis, :, :]*255.0
        reg_j = regions_det[reg_j_idx][np.newaxis, :, :]*255.0
        rgb = (np.array(rgb_im)/255.0).transpose((2, 0, 1))
        imgs = np.concatenate((rgb, reg_i, reg_j))

        # # DEBUG SAMPLE
        # deb_arr = np.array(rgb_im)
        # inds = np.array(np.where(reg_i[0, :, :]==255))
        # deb_arr[inds[0, :], inds[1, :], :] = [0, 0, 255]
        # inds = np.array(np.where(reg_j[0, :, :]==255))
        # deb_arr[inds[0, :], inds[1, :], :] = [255, 0, 0]
        # deb = Image.fromarray(deb_arr.astype('uint8'))
        # dr = ImageDraw.Draw(deb)
        # plt.imshow(deb.resize((512, 512)))
        # plt.show()

        if self.transforms:
            imgs, _ = self.transforms(imgs, None)
            # print(torch.max(imgs[:3, :, :]))
            # print(torch.min(imgs[:3, :, :]))
            
                
        # return the image, the boxlist and the idx in your dataset
        return imgs, None, idx

    def __len__(self):
        return len(self.sample_list)

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": 256, "width": 256}

    def augment(self, im, graph, rot, flip):

        # augment graph
        graph_aug = dict()
        for v1 in graph:

            # apply flip and rotation
            v1_n = self.rotate_and_flip(v1, rot, flip)
        
            # include in graph
            if v1_n not in graph_aug:
                graph_aug[v1_n] = []

            for v2 in graph[v1]:
                
                # apply flip and rotation
                v2_n = self.rotate_and_flip(v2, rot, flip)
                graph_aug[v1_n].append(v2_n)

        # augment image
        im_aug = im.rotate(rot)
        if flip == True:
            im_aug = im_aug.transpose(Image.FLIP_LEFT_RIGHT)

        return im_aug, graph_aug

    def rotate_coords(self, image_shape, xy, angle):
        org_center = (image_shape-1)/2.
        rot_center = (image_shape-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a), -org[0]*np.sin(a) + org[1]*np.cos(a)])
        return new+rot_center

    def rotate_and_flip(self, v, rot, flip):
        x, y = v
        x, y = self.rotate_coords(np.array([256, 256]), np.array([x, y]), rot)
        if flip:
            x, y = (128-abs(128-x), y) if x > 128 else (128+abs(128-x), y)
        return (x, y)

    def get_instances(self, shared_edges, rot, flip):

        masks, boxes, labels = [], [], []
        for _, _, _, c1, c2 in shared_edges:
            x1, y1 = c1
            x2, y2 = c2 
            edge = Image.new('L', (256, 256))
            dr = ImageDraw.Draw(edge)
            dr.line((x1, y1, x2, y2), fill='white', width=4)
            edge = edge.rotate(rot)
            if flip:
                edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
            edge = np.array(edge)/255.0
            inds = np.array(np.where(edge > 0.0))
            y1, x1 = np.min(inds[0, :]), np.min(inds[1, :])
            y2, x2 = np.max(inds[0, :]), np.max(inds[1, :])

            masks.append(edge)
            boxes.append([x1, y1, x2, y2])
            labels.append(1)

        if len(masks) > 0:
            masks = np.stack(masks)

        return masks, boxes, labels

    def compute_edges_mask(self, graph):
        im = Image.new('L', (256, 256))
        draw = ImageDraw.Draw(im)
        for v1 in graph:
            x1, y1 = v1
            for v2 in graph[v1]:
                x2, y2 = v2
                draw.line((x1, y1, x2, y2), width=1, fill='white')
        return np.array(im) 

    def _flood_fill(self, edge_mask, x0, y0, tag):
        new_edge_mask = np.array(edge_mask)
        nodes = [(x0, y0)]
        new_edge_mask[x0, y0] = tag
        while len(nodes) > 0:
            x, y = nodes.pop(0)
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (0 <= x+dx < new_edge_mask.shape[0]) and (0 <= y+dy < new_edge_mask.shape[0]) and (new_edge_mask[x+dx, y+dy] == 0):
                    new_edge_mask[x+dx, y+dy] = tag
                    nodes.append((x+dx, y+dy))
        return new_edge_mask

    def fill_regions(self, edge_mask):
        edge_mask = edge_mask
        tag = 2
        for i in range(edge_mask.shape[0]):
            for j in range(edge_mask.shape[1]):
                if edge_mask[i, j] == 0:
                    edge_mask = self._flood_fill(edge_mask, i, j, tag)
                    tag += 1
        return edge_mask

