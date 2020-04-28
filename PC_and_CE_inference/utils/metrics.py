import numpy as np
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw



class Metrics(): 
    def __init__(self):

        # corners
        self.curr_corner_tp = 0.0
        self.curr_corner_fp = 0.0
        self.n_corner_samples = 0.0
        self.per_corner_sample_score = {}

        # edges
        self.curr_edge_tp = 0.0
        self.curr_edge_fp = 0.0
        self.n_edge_samples = 0.0
        self.per_edge_sample_score = {}

        # loops
        self.curr_loop_tp = 0.0
        self.curr_loop_fp = 0.0
        self.n_loop_samples = 0.0
        self.per_loop_sample_score = {}

    def calc_corner_metrics(self):
        recall = self.curr_corner_tp/self.n_corner_samples
        precision = self.curr_corner_tp/(self.curr_corner_tp+self.curr_corner_fp+1e-8)
        return recall, precision

    def calc_edge_metrics(self):
        recall = self.curr_edge_tp/(self.n_edge_samples+1e-8)
        precision = self.curr_edge_tp/(self.curr_edge_tp+self.curr_edge_fp+1e-8)
        return recall, precision

    def calc_loop_metrics(self):
        recall = self.curr_loop_tp/self.n_loop_samples
        precision = self.curr_loop_tp/(self.curr_loop_tp+self.curr_loop_fp+1e-8)
        return recall, precision

    def edge_f_score(self):

        recall, precision = self.calc_edge_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        return f_score

    def print_metrics(self):

        # print scores
        values = []
        recall, precision = self.calc_corner_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        values += [precision, recall, f_score]
        print('corners - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))

        # print scores
        recall, precision = self.calc_edge_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        values += [precision, recall, f_score]        
        print('edges - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))

        recall, precision = self.calc_loop_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        values += [precision, recall, f_score]
        print('loops - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))

        return values

    def reset(self):

        # corners
        self.curr_corner_tp = 0.0
        self.curr_corner_fp = 0.0
        self.n_corner_samples = 0.0
        self.per_corner_sample_score = {}

        # edges
        self.curr_edge_tp = 0.0
        self.curr_edge_fp = 0.0
        self.n_edge_samples = 0.0
        self.per_edge_sample_score = {}

        # loops
        self.curr_loop_tp = 0.0
        self.curr_loop_fp = 0.0
        self.n_loop_samples = 0.0
        self.per_loop_sample_score = {}
        return

    def forward(self, graph_gt, junctions, juncs_on, lines_on, _id, thresh=8.0, iou_thresh=0.7):

        ## Compute corners precision/recall
        gts = np.array([list(x) for x in graph_gt])

        if len(juncs_on) > 0:
            dets = np.array(junctions)[juncs_on]
        else:
            dets = np.array([])

        per_sample_corner_tp = 0.0
        per_sample_corner_fp = 0.0
        found = [False] * gts.shape[0]
        c_det_annot = {}

        # for each corner detection
        for i, det in enumerate(dets):

            # get closest gt
            near_gt = [0, 9999999.0, (0.0, 0.0)]
            for k, gt in enumerate(gts):
                dist = np.linalg.norm(gt-det)
                if dist < near_gt[1]:
                    near_gt = [k, dist, gt] 

            # hit (<= thresh) and not found yet 
            if near_gt[1] <= thresh and not found[near_gt[0]]:
                per_sample_corner_tp += 1.0
                found[near_gt[0]] = True
                c_det_annot[juncs_on[i]] = near_gt[0]

            # not hit or already found
            else:
                per_sample_corner_fp += 1.0

        # update counters for corners
        #print('corner: ', per_sample_corner_tp, per_sample_corner_fp, gts.shape[0])
        self.curr_corner_tp += per_sample_corner_tp
        self.curr_corner_fp += per_sample_corner_fp
        self.n_corner_samples += gts.shape[0]
        self.per_corner_sample_score.update({_id: {'recall': per_sample_corner_tp/gts.shape[0], 'precision': per_sample_corner_tp/(per_sample_corner_tp+per_sample_corner_fp+1e-8)}}) 

        ## Compute edges precision/recall
        per_sample_edge_tp = 0.0
        per_sample_edge_fp = 0.0
        edge_corner_annots = edges_from_annots(graph_gt)

        # for each detected edge
        for l, e_det in enumerate(lines_on):
            c1, c2 = e_det
            
            # check if corners are mapped
            if (c1 not in c_det_annot.keys()) or (c2 not in c_det_annot.keys()):
                per_sample_edge_fp += 1.0                
                continue

            # check hit
            c1_prime = c_det_annot[c1]
            c2_prime = c_det_annot[c2]
            is_hit = False

            for k, e_annot in enumerate(edge_corner_annots):
                c3, c4 = e_annot
                if ((c1_prime == c3) and (c2_prime == c4)) or ((c1_prime == c4) and (c2_prime == c3)):
                    is_hit = True

                    # y1, x1, _, _ = building.corners_annot[c1_prime]
                    # y2, x2, _, _ = building.corners_annot[c2_prime]

                    # y3, x3 = building.corners_det[c1]
                    # y4, x4 = building.corners_det[c2]

                    # debug_im = Image.new("RGB", (256,256))
                    # draw = ImageDraw.Draw(debug_im)
                    # draw.line((x1, y1, x2, y2), "blue")
                    # draw.line((x3, y3, x4, y4), "red")
                    # plt.imshow(debug_im)
                    # plt.show()

            # hit
            if is_hit == True:
                per_sample_edge_tp += 1.0
            # not hit 
            else:
                per_sample_edge_fp += 1.0

        # update counters for edges
        #print('edge: ', per_sample_edge_tp, per_sample_edge_fp, edge_corner_annots.shape[0])
        self.curr_edge_tp += per_sample_edge_tp
        self.curr_edge_fp += per_sample_edge_fp
        self.n_edge_samples += edge_corner_annots.shape[0]
        self.per_edge_sample_score.update({_id: {'recall': per_sample_edge_tp/edge_corner_annots.shape[0], \
            'precision': per_sample_edge_tp/(per_sample_edge_tp+per_sample_edge_fp+1e-8)}}) 

        # plt.imshow(building.rgb)
        # print(building._id)
        # print(self.per_edge_sample_score[building._id])
        # print(per_sample_edge_tp)
        # print(per_sample_edge_fp)
        # print(building.edge_corner_annots.shape[0])
        # plt.show()

        ## Compute loops precision/recall
        per_sample_loop_tp = 0.0
        per_sample_loop_fp = 0.0

        # print('DETS')
        # print(lines_on, junctions) 
        # print('ANNOTS')
        # print(edge_corner_annots, np.array([list(x) for x in graph_gt]))

        corners_annots = np.array([list(x) for x in graph_gt])
        pred_edge_map = draw_edges(lines_on, junctions)
        pred_edge_map = fill_regions(pred_edge_map)
        annot_edge_map = draw_edges(edge_corner_annots, corners_annots)
        annot_edge_map = fill_regions(annot_edge_map)

        pred_rs = extract_regions(pred_edge_map)
        annot_rs = extract_regions(annot_edge_map)

        # for each predicted region
        found = [False] * len(annot_rs)
        for i, r_det in enumerate(pred_rs):

            # get closest gt
            near_gt = [0, 0, (0.0, 0.0)]
            for k, r_gt in enumerate(annot_rs):
                iou = np.logical_and(r_gt, r_det).sum()/np.logical_or(r_gt, r_det).sum()
                #print(i, k, iou)
                if iou > near_gt[1]:
                    near_gt = [k, iou, r_gt] 

            # hit (<= thresh) and not found yet 
            if near_gt[1] >= iou_thresh and not found[near_gt[0]]:
                per_sample_loop_tp += 1.0
                found[near_gt[0]] = True
                
            # not hit or already found
            else:
                per_sample_loop_fp += 1.0
        
        # update counters for corners
        self.curr_loop_tp += per_sample_loop_tp
        self.curr_loop_fp += per_sample_loop_fp
        self.n_loop_samples += len(annot_rs)
        self.per_loop_sample_score.update({_id: {'recall': per_sample_loop_tp/len(annot_rs), 'precision': per_sample_loop_tp/(per_sample_loop_tp+per_sample_loop_fp+1e-8)}})

        return

def draw_edges(edge_corner, corners, mode="det"):

    im = Image.new('L', (256, 256))
    draw = ImageDraw.Draw(im)
    for e in edge_corner:
        c1, c2 = e
        if "annot" in mode:
            y1, x1, _, _ = corners[c1]
            y2, x2, _, _ = corners[c2]
        elif "det" in mode:
            y1, x1 = corners[c1]
            y2, x2 = corners[c2]
        draw.line((x1, y1, x2, y2), width=3, fill='white')

    # import matplotlib.pyplot as plt
    # plt.imshow(im)
    # plt.show(im)
    return np.array(im)

def edges_from_annots(graph):

    # map corners
    corner_set = []
    corner_xy_map = {}
    for k, v in enumerate(graph):
        x, y = v
        corner = [x, y]
        corner_set.append(corner)
        corner_xy_map[(x, y)] = k

    # map edges
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
    return np.array(edge_set)

def extract_regions(region_mask):
    inds = np.where((region_mask > 1) & (region_mask < 255))
    tags = set(region_mask[inds])
    tag_depth = dict()
    rs = []
    for t in tags:
        if t > 0:
            r = np.zeros_like(region_mask)
            inds = np.where(region_mask == t)
            r[inds[1], inds[0]] = 1
            if r[0][0] == 0 and r[0][-1] == 0 and r[-1][0] == 0 and r[-1][-1] == 0:
                rs.append(r)
                pass
    return rs

def fill_regions(edge_mask):
    edge_mask = edge_mask
    tag = 2
    for i in range(edge_mask.shape[0]):
        for j in range(edge_mask.shape[1]):
            if edge_mask[i, j] == 0:
                edge_mask = _flood_fill(edge_mask, i, j, tag)
                tag += 1
    return edge_mask

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


