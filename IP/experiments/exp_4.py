import matplotlib.pyplot as plt
import random
import svgwrite
from PIL import Image, ImageDraw
from utils.utils import *
from utils.optimizer import reconstructBuilding
import numpy as np
import os

# Experiment IV - Enforcing closed polygons
def run_experiment_4(cs, cs_c, edge_map, th, th_c, metric, graph_annot, region_mks, rgb_dir, _id):

	# Run experiment
	junctions, juncs_on, lines_on, regs_sm_on = reconstructBuilding(cs, edge_map,
																	use_junctions_with_var=True,
																	thetas=th,
																	angle_thresh=5,
																	with_corner_edge_confidence=True,
																	corner_confs=cs_c,
																	theta_confs=th_c,
																	theta_threshold=0.25,
																	corner_edge_thresh=0.125,
																	edge_map_weight=10.0,
																	intersection_constraint=True,
																	post_process=True,

																	corner_min_degree_constraint=True,
																	with_corner_variables=True,
																	junctions_soft=True,
																	use_regions=True,
																	closed_region_constraint=True,
																	region_hit_threshold=0.1,
																	regions=region_mks,
																	_id=_id)

	# Draw regions
	im_path = '{}/{}.jpg'.format(rgb_dir, _id)
	reg_im = Image.fromarray(np.ones((256, 256))*255).convert('RGB')
	dr = ImageDraw.Draw(reg_im)
	for m in regs_sm_on:
		r = random.randint(0,255) ; g = random.randint(0,255) ; b = random.randint(0,255)
		dr.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 128))

	# Draw output image
	dwg = svgwrite.Drawing('../results/svg_regions/{}_4.svg'.format(_id), (128, 128))
	dwg.add(svgwrite.image.Image(os.path.abspath('./regions/{}.jpg'.format(_id)), size=(128, 128)))
	im_path = os.path.join(rgb_dir, _id + '.jpg')
	draw_building(dwg, junctions, juncs_on, lines_on)
	dwg.save()

	# Update metric
	metric.forward(graph_annot, junctions, juncs_on, lines_on, _id)

	return metric
