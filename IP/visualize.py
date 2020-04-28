
import pickle as p
import glob
import svgwrite
import os
import numpy as np
from utils.metrics import Metrics
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import torch
from utils.utils import * 

prefix = '/local-scratch2/nnauata/cities_dataset'
res_dir = '/local-scratch2/nnauata/outdoor_project/results/junc/3/15/2'
rgb_dir = '{}/rgb/'.format(prefix)
edge_dir = '{}/edge_map/'.format(prefix)
corner_dir = '{}/dets/corners/'.format(prefix)
annot_dir = '{}/annot/'.format(prefix)
region_dir = '/{}/regions_no_bkg/'.format(prefix)
shared_edges_fname = '{}/shared_edges_no_bkg.pkl'.format(prefix)

with open(shared_edges_fname, 'rb') as f:
    shared_edges = p.load(f, encoding='latin1')

with open('{}/valid_list.txt'.format(prefix)) as f:
    _ids = [x.strip() for x in f.readlines()]
        
os.makedirs('viz/PC', exist_ok=True)
os.makedirs('viz/PE', exist_ok=True)
os.makedirs('viz/PR', exist_ok=True)
os.makedirs('viz/CE', exist_ok=True)
os.makedirs('viz/RR', exist_ok=True)
for _id in _ids:

#     if _id not in ['1554691947.62']:
#         continue

    print(_id)

    # load detections
    fname = '{}/{}.jpg_5.pkl'.format(res_dir, _id)
    with open(fname, 'rb') as f:
        c = p.load(f, encoding='latin1')

    # apply non maxima supression
    cs, cs_c, th, th_c = nms(c['junctions'], c['junc_confs'], c['thetas'], c['theta_confs'], nms_thresh=8.0)

    # load edge map
    edge_map_path = '{}/{}.jpg'.format(edge_dir, _id)
    im_path = '{}/{}.jpg'.format(rgb_dir, _id)
    edge_map = np.array(Image.open(edge_map_path).convert('L'))/255.0

    # load region masks
    region_path = '{}/{}.npy'.format(region_dir, _id)
    region_mks = np.load(region_path)
    region_mks, shared_edges_per_id = filter_regions(region_mks, shared_edges, _id)

    # visualize all

    # draw PC
    draw_junctions(_id, cs, 'viz/PC/{}.svg'.format(_id), None, None)
    
    # draw PE
    draw_edges(_id, edge_map, 'viz/PE/{}.svg'.format(_id))
    
    # draw PR
    draw_regions(region_mks, _id, 'viz/PR/{}.png'.format(_id))
    
    # draw CE
    draw_junctions(_id, cs, 'viz/CE/{}.svg'.format(_id), th, th_c)
    
    # draw RR
    draw_shared_edges(im_path, shared_edges_per_id, _id, 'viz/RR/{}.png'.format(_id))
    