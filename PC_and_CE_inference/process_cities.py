import numpy as np
import pickle as p
import glob 
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

def compute_angles(v, neighbours):

	angles = []
	x, y = v
	for vn in neighbours:
		xn, yn = vn
		dx, dy = xn-x, yn-y
		dir_x, dir_y = (dx, dy)/np.linalg.norm([dx, dy])
		rad = np.arctan2(-dir_y, dir_x)
		ang = np.degrees(rad)
		if ang < 0:
			ang = (ang + 360) % 360
		angles.append((360.0-ang, None))

	return angles


def load_annot(graph):

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

    pointlines = []
    pointlines_index = []
    theta = []
    for v1 in graph:
        ls = []
        inds = []
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
            ls.append(edge)
            inds.append(edge_map[edge])

        pointlines.append([v1, ls, []])
        pointlines_index.append(inds)

        theta.append(compute_angles(v1, graph[v1]))

    return corner_set, edge_set, pointlines, pointlines_index, theta

DIR_PATH = '/home/nelson/Workspace/cities_dataset/'
with open('/home/nelson/Workspace/cities_dataset/all_list.txt') as f:
	ids = [x.strip() for x in f.readlines()]

# *.pkl  
#     |-- imagename: 	the name of the image  
#     |-- img:         the image data  
#     |-- points:      the set of points in the wireframe, each point is represented by its (x,y)-coordinates in the image  
#     |-- lines:       the set of lines in the wireframe, each line is represented by the indices of its two end-points  
#     |-- pointlines:     the set of associated lines of each point        
#     |-- pointlines_index:       line indeces of lines in 'pointlines'  
#     |-- junction:       the junction locations, derived from the 'points' and 'lines'  
#     |-- theta:      the angle values of branches of each junction                  

# fname = '/home/nelson/Workspace/building_reconstruction/working_model/wireframe/data/pointlines/00036866.pkl'
# with open(fname, 'rb') as f:
# 	# file opened
# 	c = p.load(f, encoding='latin1')

# 	im = Image.fromarray(c['img'])
# 	draw = ImageDraw.Draw(im)

	# for a, j in zip(c['theta'], c['junction']):

	# 	x, y = j
	# 	draw.ellipse((x-2, y-2, x+2, y+2), fill='red')
	# 	# a1s = []
	# 	# for (a1, a2) in a:
	# 	# 	a1s.append('(%.1f, %.1f)'%(a1, a2))

	# 	theta = [ '(%.1f, %.1f)'%(t, t_p)for t, t_p in a]
	# 	print(theta)
	# 	draw.text((x, y), '-'.join(theta), fill='white')
	# 		#draw.line((x, y, xt, yt), width=1, fill='green')

	# for e in c['lines']:

	# 	idx1, idx2 = e
	# 	x1, y1 = c['points'][idx1]
	# 	x2, y2 = c['points'][idx2]
	# 	draw.line((x1, y1, x2, y2), width=1, fill='green')

	# plt.imshow(im)
	# plt.show()

	# for pl in c['pointlines']:
	# 	print(pl)
	# 	im = Image.fromarray(c['img'])
	# 	draw = ImageDraw.Draw(im)
	# 	pt, ls, _ = pl
	# 	x, y = pt

	# 	for l in ls:
	# 		if len(l) > 0:
	# 			idx1, idx2 = l
	# 			x1, y1 = c['points'][idx1]
	# 			x2, y2 = c['points'][idx2]
	# 			draw.line((x1, y1, x2, y2), width=3, fill='red')
	# 	draw.ellipse((x-2, y-2, x+2, y+2), fill='blue')

	# for k, l_ind in enumerate(c['pointlines_index']):
	# 	pt = c['points'][k]

	# 	im = Image.fromarray(c['img'])
	# 	draw = ImageDraw.Draw(im)
	# 	ls = [c['lines'][k] for k in l_ind]
	# 	x, y = pt

	# 	for l in ls:
	# 		if len(l) > 0:
	# 			idx1, idx2 = l
	# 			x1, y1 = c['points'][idx1]
	# 			x2, y2 = c['points'][idx2]
	# 			draw.line((x1, y1, x2, y2), width=3, fill='red')
	# 	draw.ellipse((x-2, y-2, x+2, y+2), fill='blue')

	# 	plt.imshow(im)
	# 	plt.show()

junc_path = '{}/pointlines'.format(DIR_PATH)
if not os.path.exists(junc_path):
    os.makedirs(junc_path)

for _id in ids:
 
	annot_path = '{}/annot/{}.npy'.format(DIR_PATH, _id)
	im_path = '{}/rgb/{}.jpg'.format(DIR_PATH, _id)
	annot = np.load(open(annot_path, 'rb'), encoding='bytes')
	graph = dict(annot[()])

	im = np.array(Image.open(im_path))
	points, lines, pointlines, pointlines_index, theta = load_annot(graph)
	junction = np.array(points)

	c = {'img':im, 'imgname':_id, 'points': points, 'junction': junction, 'lines': lines, 'theta': theta, 'pointlines': pointlines, 'pointlines_index': pointlines_index}
	with open('{}//pointlines/{}.pkl'.format(DIR_PATH, _id), 'wb') as handle:
		p.dump(c, handle, protocol=p.HIGHEST_PROTOCOL)

	# # DEBUG
	# im = Image.fromarray(c['img'])
	# draw = ImageDraw.Draw(im)
	# for a, j in zip(c['theta'], c['junction']):

	# 	x, y = j
	# 	draw.ellipse((x-2, y-2, x+2, y+2), fill='red')
	# 	# a1s = []
	# 	# for (a1, a2) in a:
	# 	# 	a1s.append('(%.1f, %.1f)'%(a1, a2))

	# 	theta = [ '%.1f'%(t) for t, _ in a]
	# 	print(theta)
	# 	draw.text((x, y), '-'.join(theta), fill='white')
	# 		#draw.line((x, y, xt, yt), width=1, fill='green')

	# for e in c['lines']:

	# 	idx1, idx2 = e
	# 	x1, y1 = c['points'][idx1]
	# 	x2, y2 = c['points'][idx2]
	# 	draw.line((x1, y1, x2, y2), width=1, fill='green')

	# plt.imshow(im)
	# plt.show()

	# for e in c['lines']:

	# 	idx1, idx2 = e
	# 	x1, y1 = c['points'][idx1]
	# 	x2, y2 = c['points'][idx2]
	# 	draw.line((x1, y1, x2, y2), width=3, fill='red')
	# for pl in c['pointlines']:
	# 	print(pl)
	# 	im = Image.fromarray(c['img'])
	# 	draw = ImageDraw.Draw(im)
	# 	pt, ls, _ = pl
	# 	x, y = pt

	# 	for l in ls:
	# 		if len(l) > 0:
	# 			idx1, idx2 = l
	# 			x1, y1 = c['points'][idx1]
	# 			x2, y2 = c['points'][idx2]
	# 			draw.line((x1, y1, x2, y2), width=3, fill='red')
	# 	draw.ellipse((x-2, y-2, x+2, y+2), fill='blue')

	# 	plt.imshow(im)
	# 	plt.show()
	# print(pointlines_index)
	# for k, l_ind in enumerate(c['pointlines_index']):
	# 	pt = c['points'][k]

	# 	im = Image.fromarray(c['img'])
	# 	draw = ImageDraw.Draw(im)
	# 	ls = [c['lines'][k] for k in l_ind]
	# 	x, y = pt

	# 	for l in ls:
	# 		if len(l) > 0:
	# 			idx1, idx2 = l
	# 			x1, y1 = c['points'][idx1]
	# 			x2, y2 = c['points'][idx2]
	# 			draw.line((x1, y1, x2, y2), width=3, fill='red')
	# 	draw.ellipse((x-2, y-2, x+2, y+2), fill='blue')

	# 	plt.imshow(im)
	# 	plt.show()
