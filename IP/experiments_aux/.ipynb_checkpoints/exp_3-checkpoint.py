import matplotlib.pyplot as plt
import random
import svgwrite
from PIL import Image, ImageDraw
from utils.utils import *
from utils.optimizer import reconstructBuilding

# Experiment III - Maximizing weighted sum of junction directions
def run_experiment_3(cs, cs_c, edge_map, th, th_c, metric, graph_annot, rgb_dir, _id):

	# Run experiment
	junctions, juncs_on, lines_on = reconstructBuilding(cs, edge_map,
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
																post_process=True)

	# Draw output image
	dwg = svgwrite.Drawing('../results/svg_regions/{}_3.svg'.format(_id), (128, 128))
	im_path = os.path.join(rgb_dir, _id + '.jpg')
	draw_building(dwg, junctions, juncs_on, lines_on)
	dwg.save()

	# Update metric
	metric.forward(graph_annot, junctions, juncs_on, lines_on, _id)

	return metric