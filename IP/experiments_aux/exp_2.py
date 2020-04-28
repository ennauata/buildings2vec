import matplotlib.pyplot as plt
import random
import svgwrite
from PIL import Image, ImageDraw
from utils.utils import *
from utils.optimizer import reconstructBuilding

# Experiment II - Weights as combined junctions and edges confidence
def run_experiment_2(cs, cs_c, edge_map, th, th_c, metric, graph_annot, rgb_dir, _id):

	# Run experiment
	junctions, juncs_on, lines_on = reconstructBuilding(cs, edge_map,
																coner_to_edge_constraint=True, # corner-to-edge
																thetas=th, # corner-to-edge
																angle_thresh=5, # corner-to-edge
																with_corner_edge_confidence=True, # corners + edges
																corner_edge_thresh=0.125, # corners + edges        
																edge_map_weight=10.0, # edges           
                                                        
																corner_confs=cs_c, # corners
																theta_confs=th_c, # corner-to-edge
																theta_threshold=0.25, # corner-to-edge

																intersection_constraint=True, # intersection 
																with_corner_variables=True, # connectivity
																corner_min_degree_constraint=True, # degree
																post_process=True)

	# Draw output image
	dwg = svgwrite.Drawing('../results/svg_regions/{}_2.svg'.format(_id), (128, 128))
	im_path = os.path.join(rgb_dir, _id + '.jpg')
	draw_building(dwg, junctions, juncs_on, lines_on)
	dwg.save()

	# Update metric
	metric.forward(graph_annot, junctions, juncs_on, lines_on, _id)

	return metric

