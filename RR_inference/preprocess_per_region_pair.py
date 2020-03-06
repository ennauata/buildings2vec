import numpy as np
import pickle as p
import os
from PIL import Image, ImageDraw, ImageFilter
import random

def getIntersection(region_map, j1, j2):
    x1, y1 = j1
    x2, y2 = j2
    m = Image.new('L', (256, 256))
    dr = ImageDraw.Draw(m)
    dr.line((x1, y1, x2, y2), width=1, fill='white')
    m = np.array(m)/255.0
    inds = np.array(np.where(np.array(m) > 0.0))

    # # DEBUG
    # deb = Image.fromarray(region_map*255.0).convert('RGB')
    # dr = ImageDraw.Draw(deb) 
    # dr.line((x1, y1, x2, y2), fill='red', width=1)
    # print(np.logical_and(region_map, m).sum()/inds.shape[1])
    # plt.imshow(deb)
    # plt.show()
    return np.logical_and(region_map, m).sum()/inds.shape[1]

def map_corners_and_edges(graph):
    
    # map annotation corners
    corner_set = []
    corner_xy_map = {}
    for k, v in enumerate(graph):
        x, y = v
        corner = [x, y]
        corner_set.append(corner)
        corner_xy_map[(x, y)] = k

    # prepare edge instances for this image
    edge_set = []
    edge_map = {}
    count = 0
    for v1 in graph:
        for v2 in graph[v1]:
            x1, y1 = v1
            x2, y2 = v2
            # make an order
            if x1 > x2:
                x1, x2, y1, y2 = x2, x1, y2, y1  # swap
            elif x1 == x2 and y1 > y2:
                x1, x2, y1, y2 = x2, x1, y2, y1  # swap
            else:
                pass

            edge = (corner_xy_map[(x1, y1)], corner_xy_map[(x2, y2)])
            if edge not in edge_map:
                edge_map[edge] = count
                edge_set.append(edge)
                count += 1
    edge_set = np.array([list(e) for e in edge_set])

    return corner_set, edge_set

def regions_from_annots(graph):
    edge_mask = compute_edges_mask(graph)
    region_mask = fill_regions(edge_mask)
    masks, boxes, labels = [], [], []
    inds = np.where((region_mask >= 2) & (region_mask < 255))
    tags = set(region_mask[inds])
    for t in tags:
        m = np.zeros((256, 256))
        inds = np.array(np.where(region_mask == t))
        m[inds[0, :], inds[1, :]] = 1.0
        masks.append(m)
    masks = np.stack(masks)

    # compute corners in regions
    ms_cs = []
    cs = np.array([[x, y] for x, y in graph.keys()])
    for m in masks:
        m_cs = []
        for k, pt in enumerate(cs):
            x, y = pt
            corner_im = Image.new('L', (256, 256))
            dr = ImageDraw.Draw(corner_im)
            dr.ellipse((x-2, y-2, x+2, y+2), fill='white')
            intersect = np.logical_and(corner_im, m).sum()
            if intersect > 0:
                m_cs.append((k, x, y))

        # # DEBUG            
        # import matplotlib.pyplot as plt
        # deb = Image.fromarray(m * 255.0).convert('RGB')
        # dr = ImageDraw.Draw(deb)
        # for k, x, y in m_cs:
        #     dr.ellipse((x-2, y-2, x+2, y+2), fill='green')
        # plt.imshow(deb)
        # plt.show()

        ms_cs.append(m_cs)
    ms_cs = np.array(ms_cs)
    return masks, ms_cs

def load_annot(graph, rot, flip):

    # prepare edge instances for this image
    edge_set = set()
    for v1 in graph:
        for v2 in graph[v1]:
            x1, y1 = v1
            x2, y2 = v2
            # make an order
            if x1 > x2:
                x1, x2, y1, y2 = x2, x1, y2, y1  # swap
            elif x1 == x2 and y1 > y2:
                x1, x2, y1, y2 = x2, x1, y2, y1  # swap
            else:
                pass

            x1, y1  = rotate_coords(np.array([256, 256]), np.array([x1, y1]), rot)
            x2, y2  = rotate_coords(np.array([256, 256]), np.array([x2, y2]), rot)
            if flip:
                x1, y1 = (128-abs(128-x1), y1) if x1 > 128 else (128+abs(128-x1), y1)
                x2, y2 = (128-abs(128-x2), y2) if x2 > 128 else (128+abs(128-x2), y2)
            edge = (y1, x1, y2, x2)
            edge_set.add(edge)
    edge_set = np.array([list(e) for e in edge_set])

    corner_set = []
    for v in graph:
        x, y = v
        x, y  = rotate_coords(np.array([256, 256]), np.array([x, y]), rot)
        if flip:
            x, y = (128-abs(128-x), y) if x > 128 else (128+abs(128-x), y)
        corner = [y, x, -1, -1]
        corner_set.append(corner)

    return corner_set, edge_set

def rotate_coords(image_shape, xy, angle):
    org_center = (image_shape-1)/2.
    rot_center = (image_shape-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a), -org[0]*np.sin(a) + org[1]*np.cos(a)])
    return new+rot_center


def compute_edges_mask(graph):
    im = Image.new('L', (256, 256))
    draw = ImageDraw.Draw(im)
    for v1 in graph:
        x1, y1 = v1
        for v2 in graph[v1]:
            x2, y2 = v2
            draw.line((x1, y1, x2, y2), width=1, fill='white')
    return np.array(im) 

def _flood_fill(edge_mask, x0, y0, tag):
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

def fill_regions(edge_mask):
    edge_mask = edge_mask
    tag = 2
    for i in range(edge_mask.shape[0]):
        for j in range(edge_mask.shape[1]):
            if edge_mask[i, j] == 0:
                edge_mask = _flood_fill(edge_mask, i, j, tag)
                tag += 1
    return edge_mask

def assign_regions(regions_annot, regions_det, thresh=0.8):
    reg_det_annot = {}
    for i, reg_d in enumerate(regions_det):
        for j, reg_a in enumerate(regions_annot):
            iou = np.logical_and(reg_d, reg_a).sum()/np.logical_or(reg_d, reg_a).sum()
            if iou > thresh:
                reg_det_annot[i] = j
    return reg_det_annot

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

######## PREPROCESS DATA ########
dataset_folder = '/home/nelson/Workspace/cities_dataset'
annots_folder = '{}/annot'.format(dataset_folder)
corners_folder = '/home/nelson/Workspace/building_reconstruction/working_model/wireframe/result/junc/3/15/2'
region_folder = '{}/regions_with_bkg'.format(dataset_folder)
rgb_folder = '{}/rgb'.format(dataset_folder)


with open('/home/nelson/Workspace/cities_dataset/train_list.txt') as f:
    train_list = [line.strip() for line in f.readlines()]

samples = []
labels = np.array([])
for _id in train_list:

    print(_id)

    # annots
    annot_path = os.path.join(annots_folder, _id +'.npy')
    annot = np.load(open(annot_path, 'rb'), encoding='bytes')
    graph = dict(annot[()])

    # image
    rgb_path = os.path.join(rgb_folder, _id +'.jpg')
    rgb = Image.open(rgb_path)
    rot = 0
    flip = False
    use_gt = False

    # load annots
    corners_annot, edges_annot = map_corners_and_edges(graph)
    regions_annot, region_corners_annots = regions_from_annots(graph)

    # load detections
    fname = '{}/{}.jpg_5.pkl'.format(corners_folder, _id)
    with open(fname, 'rb') as f:
        c = p.load(f, encoding='latin1')

    # apply non maxima supression
    corners_det, _, _, _ = nms(c['junctions'], c['junc_confs'], c['thetas'], c['theta_confs'], nms_thresh=8.0)

    # load regions
    region_path = '{}/{}.npy'.format(region_folder, _id)
    regions_det = np.load(region_path)

    # match regions with annotation 
    region_det_annot = assign_regions(regions_annot, regions_det)

    # compute per region info
    sm_rgb = np.array(rgb.resize((64, 64)))/255.0
    sm_rgb = sm_rgb.transpose(2, 0, 1)
    for i in region_det_annot:
        reg_det_i = regions_det[i]
        reg_annot_i = regions_annot[region_det_annot[i]]
        sm_reg_i = np.array(Image.fromarray(reg_det_i*255.0).resize((64, 64)))/255.0

        reg_info = {'id':_id, 'imgname':_id, 'sm_reg': sm_reg_i, 'reg_det':reg_det_i, \
                    'reg_annot':reg_annot_i, 'rgb':rgb, 'sm_rgb': sm_rgb}

        with open('{}/{}_{}.pkl'.format(region_folder, _id, i), 'wb') as handle:
            p.dump(reg_info, handle, protocol=p.HIGHEST_PROTOCOL)

    # if _id not in '1554523086.76':
    #     continue

    region_pairs = []
    for i in region_det_annot:
        for j in region_det_annot:
            if i > j:

                # compute intersecting corners between two regions - positive samples
                annot_corner_i = region_corners_annots[region_det_annot[i]]
                annot_corner_j = region_corners_annots[region_det_annot[j]]
                reg_annot_i = regions_annot[region_det_annot[i]]
                reg_annot_j = regions_annot[region_det_annot[j]]
                if i == len(region_det_annot)-1:
                    reg_annot_i = 1.0-reg_annot_i
                    reg_annot_i = Image.fromarray(reg_annot_i*255.0)
                    reg_annot_i = reg_annot_i.filter(ImageFilter.MinFilter(3))
                    reg_annot_i = np.array(reg_annot_i)/255.0

                # assign edges to region i
                annot_edge_i = []
                for k1, x1, y1 in annot_corner_i:
                    for k2, x2, y2 in annot_corner_i:
                        for k3, k4 in edges_annot:
                            if k1 > k2:
                                if ((k1 == k3) or (k1 == k4)) and ((k2 == k3) or (k2 == k4)):
                                    if getIntersection(reg_annot_i, (x1, y1), (x2, y2)) < 0.1:
                                        p1 = (x1, y1)
                                        p2 = (x2, y2)
                                        annot_edge_i.append((k1, k2, p1, p2))

                # # DEBUG
                # import matplotlib.pyplot as plt
                # deb_edge_i = Image.new('RGB', (256, 256))
                # dr = ImageDraw.Draw(deb_edge_i)
                # for k1, k2, p1, p2 in annot_edge_i:
                #     x1, y1 = p1
                #     x2, y2 = p2
                #     dr.line((x1, y1, x2, y2), fill='magenta', width=3)
                # plt.figure()
                # plt.imshow(deb_edge_i)

                # assign edges to region j
                annot_edge_j = []
                for k1, x1, y1 in annot_corner_j:
                    for k2, x2, y2 in annot_corner_j:
                        for k3, k4 in edges_annot:
                            if k1 > k2:
                                if ((k1 == k3) or (k1 == k4)) and ((k2 == k3) or (k2 == k4)):
                                    if getIntersection(reg_annot_j, (x1, y1), (x2, y2)) < 0.1:
                                        p1 = (x1, y1)
                                        p2 = (x2, y2)
                                        annot_edge_j.append((k1, k2, p1, p2))
                # # DEBUG
                # deb_edge_j = Image.new('RGB', (256, 256))
                # dr = ImageDraw.Draw(deb_edge_j)
                # for k1, k2, p1, p2 in annot_edge_j:
                #     x1, y1 = p1
                #     x2, y2 = p2
                #     dr.line((x1, y1, x2, y2), fill='magenta', width=3)
                # plt.figure()
                # plt.imshow(deb_edge_j)
                # plt.show()

                # DEBUG - display region pairs
                reg_det_i = regions_det[i]
                reg_det_j = regions_det[j]
                reg_det_i = Image.fromarray(reg_det_i * 255.0)
                reg_det_j = Image.fromarray(reg_det_j * 255.0)

                deb = Image.open(rgb_path)
                dr = ImageDraw.Draw(deb)
                r = random.randint(0,255) ; g = random.randint(0,255) ; b = random.randint(0,255)
                dr.bitmap((0, 0), reg_det_i.convert('L'), fill=(r, g, b, 128))
                r = random.randint(0,255) ; g = random.randint(0,255) ; b = random.randint(0,255)
                dr.bitmap((0, 0), reg_det_j.convert('L'), fill=(r, g, b, 128))


                # Generate annotation boundary
                pos_samples_ij = []
                for l, (k1, k2, p1, p2) in enumerate(annot_edge_i):
                    for m, (k3, k4, p3, p4) in enumerate(annot_edge_j):
                        if ((k1, k2) == (k3, k4)) or ((k1, k2) == (k4, k3)) or ((k2, k1) == (k3, k4)) or ((k2, k1) == (k4, k3)):
                            x1, y1 = p1
                            x2, y2 = p2
                            pos_samples_ij.append((_id, l, m, p1, p2))
                            dr.line((x1, y1, x2, y2), fill='magenta', width=3)

                # # DEBUG - display region pairs
                # plt.imshow(deb)
                # plt.show()

                deb.save('{}/shared_edges/debug/{}_{}_{}_boundary.jpg'.format(dataset_folder, _id, i, j))

                # save per rergion pair info
                # deb.save('./data/pairs/{}_{}_{}_all.jpg'.format(_id, i, j))
                reg_pair_info = {'shared_edges': pos_samples_ij}
                with open('{}/shared_edges/pairs/{}_{}_{}.pkl'.format(dataset_folder, _id, i, j), 'wb') as handle:
                    p.dump(reg_pair_info, handle, protocol=p.HIGHEST_PROTOCOL)

                # labels = np.concatenate([labels, np.ones(len(pos_samples_ij)), np.zeros(len(neg_samples_ij))], -1)
                # samples += pos_samples_ij + neg_samples_ij

# save data
samples_info = {'samples':samples, 'labels':labels}
with open('{}/data.pkl'.format(dataset_folder), 'wb') as handle:
    p.dump(samples_info, handle, protocol=p.HIGHEST_PROTOCOL)
