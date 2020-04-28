import matplotlib.pyplot as plt
import random
import svgwrite
from PIL import Image, ImageDraw
from utils.utils import *
from utils.optimizer import reconstructBuilding

# Baseline - Trust on junctions orientations and optimize the number of edges
def run_experiment_0(cs, edge_map, th_filtered, metric, graph_annot, rgb_dir, _id):

	# Run experiment
	junctions, juncs_on, lines_on = reconstructBuilding(cs, edge_map,
														edge_threshold=0.5,
														with_edge_confidence=True, # edges
														edge_map_weight=10.0, # edges
														intersection_constraint=True, # intersection 
														with_corner_variables=True, # connectivity
														corner_min_degree_constraint=True, # degree
														post_process=True)

	# Draw output image
	dwg = svgwrite.Drawing('../results/svg_regions/{}_0.svg'.format(_id), (128, 128))
	im_path = os.path.join(rgb_dir, _id + '.jpg')
	draw_building(dwg, junctions, juncs_on, lines_on)
	dwg.save()

	# Update metric
	metric.forward(graph_annot, junctions, juncs_on, lines_on, _id)

	return