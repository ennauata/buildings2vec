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

class BuildingsDataset(object):
    def __init__(self, img_dir, ann_file, id_file, transforms=None, split='train'):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.split = split
        self.building_id_file = id_file
        with open(self.building_id_file) as f:
            self.building_ids = [x.strip() for x in f.readlines()]
        self.building_ids = self.building_ids
        self.regions_folder = '/local-scratch2/nnauata/cities_dataset/shared_edges/regions'
        self.shared_edges_folder = '/local-scratch2/nnauata/cities_dataset/shared_edges/pairs'
        return

    def __getitem__(self, idx):
        
        _building_id = self.building_ids[idx]

        # load data
        rgb_im = Image.open('{}/{}.jpg'.format(self.img_dir, _building_id))
        annot_path = os.path.join('{}/{}.npy'.format(self.ann_file, _building_id))
        annot = np.load(open(annot_path, 'rb'), encoding='bytes', allow_pickle=True)
        graph = dict(annot[()])

        # augment data
#         print('{}/{}_*.pkl'.format(self.regions_folder, _building_id))
        regs_paths = glob.glob('{}/{}_*.pkl'.format(self.regions_folder, _building_id))
#         print(len(regs_paths))
        if len(regs_paths) < 2:
            return self.__getitem__((idx+1)%len(self.building_ids))

        reg_i_path, reg_j_path = random.sample(regs_paths, 2)
        reg_i_idx = reg_i_path.split(str(_building_id)+'_')[-1].replace('.pkl', '')
        reg_j_idx = reg_j_path.split(str(_building_id)+'_')[-1].replace('.pkl', '')

        # load regions info
        with open('{}/{}_{}.pkl'.format(self.regions_folder, _building_id, reg_i_idx), 'rb') as f:
            reg_i_info = p.load(f)
        with open('{}/{}_{}.pkl'.format(self.regions_folder, _building_id, reg_j_idx), 'rb') as f:
            reg_j_info = p.load(f)

        # augmentation
        rot = 0
        flip = False
        if self.split == 'train':
            rot = np.random.choice([0, 90, 180, 270])
            flip = np.random.choice([True, False])
        rgb = reg_i_info['rgb'].rotate(rot)
        reg_i = Image.fromarray(reg_i_info['reg_det']*255.0).rotate(rot)
        reg_j = Image.fromarray(reg_j_info['reg_det']*255.0).rotate(rot)
        if flip == True:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            reg_i = reg_i.transpose(Image.FLIP_LEFT_RIGHT)
            reg_j = reg_j.transpose(Image.FLIP_LEFT_RIGHT)
        rgb = (np.array(rgb_im)/255.0).transpose((2, 0, 1))
        reg_i = (np.array(reg_i))[np.newaxis, :, :]
        reg_j = (np.array(reg_j))[np.newaxis, :, :]
        imgs = np.concatenate((rgb, reg_i, reg_j))

        # print(np.max(rgb))
        # print(np.max(reg_i))
        # print(np.max(reg_j))
        
        # generate edges masks
        try:
            with open('{}/{}_{}_{}.pkl'.format(self.shared_edges_folder, _building_id, reg_i_idx, reg_j_idx), 'rb') as f:
                pair_info = p.load(f)
        except:
            with open('{}/{}_{}_{}.pkl'.format(self.shared_edges_folder, _building_id, reg_j_idx, reg_i_idx), 'rb') as f:
                pair_info = p.load(f)

        shared_edges = pair_info['shared_edges']
        if len(shared_edges) == 0:
            return self.__getitem__((idx+1)%len(self.building_ids))

        # generate regions
        masks, boxes, labels = self.get_instances(shared_edges, rot, flip)

        # convert to tensor
        masks = torch.tensor(masks)
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)


        # # DEBUG SAMPLE
        # for m, bb in zip(masks, boxes):

        #     print(reg_i.shape)
        #     print(reg_j.shape)
        #     reg_i_im = Image.fromarray(reg_i[0, :, :]*255)
        #     reg_j_im = Image.fromarray(reg_j[0, :, :]*255)

        #     deb_arr = np.zeros((256, 256, 3))
        #     inds = np.array(np.where(reg_i[0, :, :]==1))
        #     deb_arr[inds[0, :], inds[1, :], :] = [0, 0, 255]
        #     inds = np.array(np.where(reg_j[0, :, :]==1))
        #     deb_arr[inds[0, :], inds[1, :], :] = [255, 0, 0]
        #     print(m.shape)
        #     inds = np.array(np.where(m[:, :]==1))
        #     deb_arr[inds[0, :], inds[1, :], :] = [0, 255, 0]
        #     deb = Image.fromarray(deb_arr.astype('uint8'))
        #     dr = ImageDraw.Draw(deb)
        #     print(list(bb))
        #     x0, y0 = bb[:2]
        #     x1, y1 = bb[2:]
        #     print(bb)
        #     dr.line([(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)], fill='green', width=2)
        #     #dr.line([(y0, x0), (y0, x1), (y1, x1), (y1, x0), (y0, x0)], fill='green', width=2)

        #     plt.figure()
        #     plt.imshow(deb.resize((512, 512)))
            
        #     plt.figure()
        #     plt.imshow(reg_i_im)

        #     plt.figure()
        #     plt.imshow(reg_j_im)

        #     plt.show()


        # create masks object
        masks = SegmentationMask(masks, rgb_im.size, mode='mask')

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, rgb_im.size, mode="xyxy")

        # add to the boxlist
        boxlist.add_field("labels", labels)
        boxlist.add_field("masks", masks)

        if self.transforms:
            imgs, boxlist = self.transforms(imgs, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return imgs, boxlist, idx

    def __len__(self):
        return len(self.building_ids)

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

