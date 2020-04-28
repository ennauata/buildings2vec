import matplotlib.pyplot as plt
import random
import svgwrite
from PIL import Image, ImageDraw
from utils.utils import *
from utils.optimizer import reconstructBuilding

# Experiment I - Weighted edges indicator variables using edges confidence
def run_experiment_1(cs, cs_c, edge_map, th_filtered, metric, graph_annot, rgb_dir, _id):

	junctions, juncs_on, lines_on = reconstructBuilding(cs, edge_map,
																corner_confs=cs_c,
																with_corner_edge_confidence=True, # corners + edges
																corner_edge_thresh=0.125, # corners + edges
																edge_map_weight=10.0, # edges
																intersection_constraint=True, # intersection 
																with_corner_variables=True, # connectivity
																corner_min_degree_constraint=True, # degree
																post_process=True)

    

                                
	# Draw output image
	dwg = svgwrite.Drawing('../results/svg_regions/{}_1.svg'.format(_id), (128, 128))
	im_path = os.path.join(rgb_dir, _id + '.jpg')
	draw_building(dwg, junctions, juncs_on, lines_on)
	dwg.save()

	# Update metric
	metric.forward(graph_annot, junctions, juncs_on, lines_on, _id)

	return