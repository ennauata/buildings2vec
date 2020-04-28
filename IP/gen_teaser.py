
import pickle as p
import glob
import svgwrite
import os
import numpy as np
from utils.metrics import Metrics
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from models.graph import EdgeClassifier
from models.resnet import resnet152, resnet18
import torch
from utils import * 
from experiments.exp_0 import *
from experiments.exp_1 import *
from experiments.exp_2 import *
from experiments.exp_3 import *
from experiments.exp_4 import *
from experiments.exp_5 import *
from experiments.exp_6 import *

prefix = '/local-scratch2/nnauata/for_teaser'
res_dir = '/local-scratch2/nnauata/outdoor_project/result/junction_detector/3/15/2'
rgb_dir = '{}/rgb/'.format(prefix)
edge_dir = '{}/edge_map/'.format(prefix)
annot_dir = '{}/annot/'.format(prefix)
region_dir = '/{}/regions_no_bkg/'.format(prefix)
shared_edges_fname = '{}/shared_edges_no_bkg.pkl'.format(prefix)

with open(shared_edges_fname, 'rb') as f:
    shared_edges = p.load(f, encoding='latin1')

with open('{}/all_list.txt'.format(prefix)) as f:
	_ids = [x.strip() for x in f.readlines()]

for _id in _ids:

# 	if _id not in ['1554148701.17']:
# 		continue

	print(_id)

	# load detections
	fname = '{}/{}.jpg_5.pkl'.format(res_dir, _id)
	with open(fname, 'rb') as f:
		c = p.load(f, encoding='latin1')

	# apply non maxima supression
	cs, cs_c, th, th_c = nms(c['junctions'], c['junc_confs'], c['thetas'], c['theta_confs'], nms_thresh=8.0)

	# load annotations
	p_path = '{}/{}.npy'.format(annot_dir, _id)
	v_set = np.load(open(p_path, 'rb'),  encoding='bytes', allow_pickle=True)
	graph_annot = dict(v_set[()])[b'graph']
	cs_annot, es_annot = load_annots(graph_annot)

	# load edge map
	edge_map_path = '{}/{}.jpg'.format(edge_dir, _id)
	im_path = '{}/{}.jpg'.format(rgb_dir, _id)
	edge_map = np.array(Image.open(edge_map_path).convert('L'))/255.0

	# load region masks
	region_path = '{}/{}.npy'.format(region_dir, _id)
	region_mks = np.load(region_path)
	region_mks, shared_edges_per_id = filter_regions(region_mks, shared_edges, _id)

	# compute edge scores from classifier
	lw_from_cls = None #get_edge_scores(cs, region_mks, rgb_dir, _id)

	# draw corners
	dwg = svgwrite.Drawing('../results/for_teaser/{}_7.svg'.format(_id), (128, 128))
	im_path = os.path.join(rgb_dir, _id + '.jpg')
	draw_building(dwg, cs, np.array(range(cs.shape[0])), [])
	dwg.save()

	# draw annotations
	dwg = svgwrite.Drawing('../results/for_teaser/{}_8.svg'.format(_id), (128, 128))
	im_path = os.path.join(rgb_dir, _id + '.jpg')
	draw_building(dwg, cs_annot, np.array(range(cs_annot.shape[0])), es_annot)
	dwg.save()

	# filter angles
	th_filtered, th_c_filtered = filter_angles(th, th_c, thresh=0.5)

   ################### Run Experiments ####################

# 	run_experiment_0(cs, edge_map, th_filtered, metrics[0], graph_annot, rgb_dir, _id)
# 	run_experiment_1(cs, edge_map, th_filtered, metrics[1], graph_annot, rgb_dir, _id)
# 	run_experiment_2(cs, cs_c, edge_map, th_filtered, metrics[2], graph_annot, rgb_dir, _id)
# 	run_experiment_3(cs, cs_c, edge_map, th, th_c, metrics[3], graph_annot, rgb_dir, _id)
# 	run_experiment_4(cs, cs_c, edge_map, th, th_c, metrics[4], graph_annot, region_mks, rgb_dir, _id)
# 	run_experiment_5(cs, cs_c, edge_map, th, th_c, metrics[5], graph_annot, region_mks, shared_edges, rgb_dir, _id)
	run_experiment_6(cs, cs_c, edge_map, th, th_c, None, graph_annot, region_mks, shared_edges, rgb_dir, _id)
# 	draw_junctions(_id, cs, th, th_c)
# 	show_shared_edges(im_path, shared_edges_per_id, _id)
    
# # print metrics
# all_results = []
# for k, m in enumerate(metrics):
# 	print('experiment %d'%(k))
# 	values =  m.print_metrics()
# 	values = [x*100.0 for x in values]
# 	all_results.append(values)
# stress(all_results)