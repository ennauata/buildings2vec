# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import cv2
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from PIL import Image, ImageDraw
from functools import reduce
import sys
import math
import operator
from rdp import rdp
from scipy.signal import find_peaks
sys.path.insert(0, '/local-scratch2/nnauata/outdoor_project/junction_detector/junc/utils/')
# from intersections import doIntersect
import pickle as p

# from optimizer import extractCycle
out_dir = '/local-scratch2/nnauata/for_teaser/regions_no_bkg'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def do_buildings_evaluation(dataset, predictions, box_only, output_folder, iou_types, expected_results, expected_results_sigma_tol):

    pred_boxlists = []
    image_list = []
    n_cols = 0
    for image_id, prediction in enumerate(predictions):

        # retrieve image information
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        
        # retrive predictions
        pred_bbox = prediction.bbox.numpy()
        pred_score = prediction.get_field("scores").numpy()
        labels = prediction.get_field("labels")
        masks = prediction.get_field("mask")

        # retrieve rgb image
        print(dataset.building_ids[image_id])
        rgb_im = Image.open('{}/{}.jpg'.format(dataset.img_dir, dataset.building_ids[image_id]))
        rgb_im = np.array(rgb_im)

        # unmask predictions
        masker = Masker(threshold=0.5, padding=1)
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0].numpy().reshape(-1, 256, 256)
        inds = np.array(np.where(pred_score>0.5))[0, :]
        masks = masks[inds, :, :]
        labels = labels[inds]
        pred_score = pred_score[inds]

        # ## DEBUG ##
        # import random
        # deb_im = Image.fromarray(rgb_im)
        # dr = ImageDraw.Draw(deb_im)
        # ## DEBUG ##

        # accumulate predictions
        mask_list = []
        # print(labels.shape, pred_score.shape, masks.shape)
        for s, l, m in zip(pred_score, labels, masks):
            if l == 1:
                mask_list.append(m)
            
        #     ## DEBUG ##
        #     print(l, s)
        #     msk = Image.fromarray((m*255.0).astype('uint8'))
        #     if l == 1:
        #         r = random.randint(0,128) ; g = random.randint(0,128) ; b = random.randint(0,128)
        #         dr.bitmap((0, 0), msk.convert('L'), fill=(r, g, b, 128))
        #     else:
        #         dr.bitmap((0, 0), msk.convert('L'), fill=(0, 0, 0, 128))
        #     ## DEBUG ##

        # plt.imshow(deb_im)
        # plt.show()
        
        np.save('{}/{}.npy'.format(out_dir, dataset.building_ids[image_id]), np.array(mask_list))
    print('FINISHED')
    return

def getAngle(pt1, pt2):
    # return angle in clockwise direction
    x, y = pt1
    xn, yn = pt2
    dx, dy = xn-x, yn-y
    dir_x, dir_y = (dx, dy)/np.linalg.norm([dx, dy])
    rad = np.arctan2(-dir_y, dir_x)
    ang = np.degrees(rad)
    if ang < 0:
        ang = (ang + 360) % 360
    return 360-ang

def doHit(j1, j2, region_map, thresh=0.2):

    edge_im = Image.new('L', (256, 256))
    dr = ImageDraw.Draw(edge_im)
    x1, y1 = j1
    x2, y2 = j2
    dr.line((x1, y1, x2, y2), width=2, fill='white')
    edge_map = np.array(edge_im)/255.0
    iou = np.logical_and(edge_im, region_map).sum()/edge_map.sum()
    if iou > thresh:
        return 1.0
    return 0

def getLineWeight(j1, j2, edge_map):

    x1, y1 = j1
    x2, y2 = j2
    edge_map = np.array(edge_map)/255.0
    m = Image.new('L', (256, 256))
    dr = ImageDraw.Draw(m)
    dr.line((x1, y1, x2, y2), width=2, fill='white')
    inds = np.array(np.where(np.array(m) > 0.0))
    weight = np.sum(edge_map[inds[0, :], inds[1, :]])/inds.shape[1]
    return weight

def getIOUWeight(poly, region_map):
    m = Image.new('L', (256, 256))
    dr = ImageDraw.Draw(m)
    poly = [tuple(xy) for xy in poly]
    dr.polygon(poly, fill='white')
    # plt.imshow(m)
    # print(poly)
    # plt.show()
    m = np.array(m)/255.0
    iou = np.logical_and(m, region_map).sum()/np.logical_or(m, region_map).sum()
    return iou

def inBetween(n, a, b):
    n = (360 + (n % 360)) % 360
    a = (3600000 + a) % 360
    b = (3600000 + b) % 360
    if (a < b):
        return a <= n and n <= b
    return a <= n or n <= b

def doSelfIntersect(j2, j3, cycle, junctions):
    
    # check intersection
    intersect = False
    j0 = junctions[cycle[0]]
    for k in cycle[1:]:
        j1 = junctions[k]
        if doIntersect(j0, j1, j2, j3):
            return True
        j0 = np.array(j1)
    return False

def _debug_print_cycle(cycle, iou, region_map, junctions):

    # debug
    cycle_im = Image.fromarray(region_map*255.0).convert('RGB')
    dr = ImageDraw.Draw(cycle_im)
    poly = np.array(cycle)
    to_draw = []
    for xy in junctions[poly]:
        to_draw.append(xy[0])
        to_draw.append(xy[1])
    to_draw += to_draw[:2]
    dr.line(to_draw, fill='green', width=2)

    for x, y in junctions[poly]:
        dr.ellipse((x-2, y-2, x+2, y+2), fill='blue')

    for k, xy in enumerate(junctions[poly]):
        x, y = xy
        dr.text((x, y), str(k), fill='red')

    # print(iou)
    # plt.imshow(cycle_im)
    # plt.show()
    return

def extractCycleContour(region_map):
    # print(region_map.shape)
    # print(np.max(region_map))
    region_map = region_map * 255
    im_reg = Image.fromarray(region_map).convert('RGB')
    dr = ImageDraw.Draw(im_reg)

    ret, thresh = cv2.threshold(region_map, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [rdp(c, epsilon=0.001) for c in contours]

    # filter contours
    best = [0.0, []]
    for c in contours:
        to_draw = []
        cycle = []
        for xy in c:
            x, y = np.array(xy).ravel()
            cycle.append((x, y))
            to_draw.append(x)
            to_draw.append(y)
        to_draw += to_draw[:2]
        dr.line(to_draw, fill='green', width=2)

        for xy in c:
            x, y = np.array(xy).ravel()
            dr.ellipse((x-2, y-2, x+2, y+2), fill='blue')

        if len(cycle) > 2:
            iou = getIOUWeight(cycle, region_map/255.0)
        else:
            iou = -1

        if iou > best[0]:
            best[0] = iou
            best[1] = cycle

    # plt.imshow(im_reg)
    # plt.show()
    return best[1]

def extractCycleGreedy(region_map, edge_map, junctions, EDGE_THRESHOLD=0.2, MAX_CYCLE_SIZE=20):

    # pre compute line weights
    lws = {}
    for k, j1 in enumerate(junctions):
        for l, j2 in enumerate(junctions):
            lw = getLineWeight(j1, j2, edge_map)
            lws[(k, l)] = lw
            lws[(l, k)] = lw

    # initialize
    all_cycles, all_ious, all_lws = [], [], []
    for k in range(junctions.shape[0]):
        for l in range(junctions.shape[0]):
            if k > l and lws[(k, l)] >= EDGE_THRESHOLD:
                all_cycles.append([k, l])
                all_ious.append([0.0, 0.0])
                all_lws.append(lws[k, l])

    for i in range(len(all_cycles)):
        for _ in range(MAX_CYCLE_SIZE):
            keep_track = [-1.0, -1]
            for k in range(junctions.shape[0]):

                j_k = junctions[k]
                j_last = junctions[all_cycles[i][-1]]
                j_first = junctions[all_cycles[i][0]]

                # check intersections
                intersect1 = doSelfIntersect(j_last, j_k, all_cycles[i], junctions)
                intersect2 = doSelfIntersect(j_k, j_first, all_cycles[i]+[k], junctions)

                _last = all_cycles[i][-1]
                # _first = all_cycles[i][0]
                # compute line weight
                if _last > -1:
                    lw_next = lws[(_last, k)] + all_lws[i]
                    #cycle_lw = (lw_next + lws[(k, _first)])/len(all_cycles[i]+[k])
                else:
                    continue

                if (k not in all_cycles[i]) and not intersect1 and not intersect2 and lws[(_last, k)] >= EDGE_THRESHOLD:
                    cycle_juncs_next = junctions[np.array(all_cycles[i]+[k])]
                    iou_weight = getIOUWeight(cycle_juncs_next, region_map)
                    if iou_weight > keep_track[0]:
                        keep_track[0] = iou_weight
                        keep_track[1] = k
            all_cycles[i].append(keep_track[1])
            all_ious[i].append(keep_track[0])
            # all_lws[i] = lw_next

            # if keep_track[0] >= 0:
            #     _debug_print_cycle(all_cycles[i], keep_track[0], region_map, junctions)


    all_cycles = np.array(all_cycles)
    all_ious = np.array(all_ious)
    best = [-1.0, None]
    for i in range(2, all_cycles.shape[1]):
        max_i = np.argmax(all_ious[:, i])
        cycle_i = all_cycles[max_i, :i+1]
        if all_ious[max_i, i] > best[0]:
            best[0] = all_ious[max_i, i]
            best[1] = cycle_i

    # debug
    # cycle_i = best[1]
    # iou_i = best[0]
    # cycle_im = Image.fromarray(region_map*255.0).convert('RGB')
    # dr = ImageDraw.Draw(cycle_im)
    # poly = np.array(cycle_i)
    # to_draw = []
    # for xy in junctions[poly]:
    #     to_draw.append(xy[0])
    #     to_draw.append(xy[1])
    # to_draw += to_draw[:2]
    # dr.line(to_draw, fill='green', width=2)
    # print(i, iou_i)
    # plt.imshow(cycle_im)
    # plt.show()

    # print(np.array(all_cycles).shape)
    # print(np.array(all_ious).shape)

    return best[1]

def extractCycleBFS(region_map, edge_map, junctions, MAX_NUM_NODES=1000, MAX_DEPTH=10):

    # pre compute line weights]
    lws = {}
    for k, j1 in enumerate(junctions):
        for l, j2 in enumerate(junctions):
            if l > k:
                lw = getLineWeight(j1, j2, edge_map)
                lws[(k, l)] = lw
                lws[(l, k)] = lw

    best_node = [[], 0.0]
    nodes_to_expand = [(x, [x], 0.0, 0.0) for x in range(len(junctions))] # [curr_j, cycle, lw]
    while len(nodes_to_expand) > 0:
        node = nodes_to_expand.pop()
        _curr = node[0]
        j_curr = junctions[_curr]
        cycle_curr = node[1]
        w_curr = node[2]
        w_tot_curr = node[3]

        # keep track of best node
        if len(cycle_curr) >= 3:

            w_tot_best = best_node[1]
            j_last = junctions[cycle_curr[-1]]
            j_first = junctions[cycle_curr[0]]
            intersect = doSelfIntersect(j_first, j_last, cycle_curr, junctions)

            if w_tot_curr > w_tot_best and not intersect:
                best_node[0] = cycle_curr
                best_node[1] = w_tot_curr

                # iou greater than .9
                if best_node[1] > 0.9:
                    break

        # too many nodes to expand
        if (len(nodes_to_expand) > MAX_NUM_NODES) or (len(cycle_curr)+1 >= MAX_DEPTH):
            continue 

        # search neighbours
        js_next = [(k, x) for k, x in enumerate(junctions) if k not in cycle_curr]
        for _next, j_next in js_next:

            # prunning
            in_between = True
            if len(cycle_curr)>1:

                # get angle ref
                inds = np.argwhere(region_map==1)
                rc = (inds.sum(0)/inds.shape[0])[::-1]
                a_ref = getAngle(j_curr, rc)

                _prev = cycle_curr[-2]
                j_prev = junctions[_prev]
                a_cn = getAngle(j_curr, j_next)
                a_cp = getAngle(j_curr, j_prev)

                in_between = inBetween(a_cn, a_cp, a_ref)

            # check intersection
            intersect = doSelfIntersect(j_curr, j_next, cycle_curr, junctions)
            w_edge = lws[(_curr, _next)]
            if in_between and not intersect and w_edge > 0.1:
                w_next = w_edge + w_curr
                if len(cycle_curr) >= 2:
                    _first = cycle_curr[0]
                    lw = lws[(_first, _next)]
                    lw_cycle = (w_next + lw)/(len(cycle_curr)+1)
                    next_cycles_juncs = junctions[np.array(cycle_curr+[_next])]
                    iou_weight = getIOUWeight(next_cycles_juncs, region_map)
                    w_tot_next = iou_weight #*lw_cycle
                else:
                    w_tot_next = 0.0

                cyle_next = list(cycle_curr+[_next])
                # if (len(nodes_to_expand) < MAX_NUM_NODES) and (len(cyle_next) < MAX_DEPTH): 
                nodes_to_expand.append([_next, cyle_next, w_next, w_tot_next])

        # sort list
        nodes_to_expand.sort(key=lambda x: x[3], reverse=False)
        
    return best_node[0]
    # print(best_node)  

    # deb = Image.fromarray(region_map*255.0).convert('RGB')
    # dr = ImageDraw.Draw(deb)
    # poly = np.array(best_node[0])
    # to_draw = []
    # for xy in junctions[poly]:
    #     to_draw.append(xy[0])
    #     to_draw.append(xy[1])
    # to_draw += to_draw[:2]
    # dr.line(to_draw, fill='green', width=3)
    # for k, xy in enumerate(junctions[poly]):
    #     x, y = xy
    #     dr.text((x, y), str(k), fill='magenta')

    # plt.imshow(deb)
    # print(poly)
    # plt.show()  

def extractCycle(region_map, edge_map, junctions):
    
    for l, jk in enumerate(junctions):
        j_curr = jk
        _curr = l
        cycle = []
        print('ITER' )
        while True:
            as_j12, j2s, j2s_inds = [], [], []
            cycle.append(_curr)
            cadidates = [(k, x) for k, x in enumerate(junctions) if k not in cycle]
            #print(cadidates)

            # sort clockwise
            for k, j2 in cadidates:
                as_j12.append(getAngle(j_curr, j2))
                j2s.append(j2)
                j2s_inds.append(k)
            
            # convert to array
            as_j12 = np.array(as_j12)
            j2s = np.array(j2s)
            j2s_inds = np.array(j2s_inds)

            # sort
            inds = np.argsort(as_j12)
            as_j12 = as_j12[inds]
            j2s = j2s[inds]
            j2s_inds = j2s_inds[inds]

            # get angle ref
            inds = np.argwhere(region_map==1)
            rc = (inds.sum(0)/inds.shape[0])[::-1]
            ref = getAngle(j_curr, rc)

            # get edgness scores and find peaks
            edge_scores = [getLineWeight(j_curr, j2, edge_map) for j2 in j2s]
            edge_scores = 2*edge_scores
            peaks, _ = find_peaks(edge_scores, height=0)
            peaks = [p%(len(edge_scores)/2) for p in peaks]
            peaks = np.array(list(set(peaks))).astype('int')
            print(peaks)

            # pick closest peak to ref
            found = -1
            as_peaks = as_j12[peaks]
            for p, a in zip(peaks[::-1], as_peaks[::-1]):
                if a%360 < ref:
                    found = p
                    break

            # debug
            xc, yc = rc
            x1, y1 = j_curr
            deb = Image.fromarray(region_map*255.0).convert('RGB')
            dr = ImageDraw.Draw(deb)

            for k, j2 in enumerate(j2s):
                x2, y2 = j2
                dr.line((x1, y1, x2, y2), width=3, fill='green')
                
            for k, j in enumerate(junctions):
                x, y = j
                # if k != _curr:
                #     dr.ellipse((x-2, y-2, x+2, y+2), fill='red')
                # else:
                #     dr.ellipse((x-2, y-2, x+2, y+2), fill='blue')
                dr.text((x, y), str(k), fill='magenta')

            dr.ellipse((x1-2, y1-2, x1+2, y1+2), fill='blue')
            dr.ellipse((xc-2, yc-2, xc+2, yc+2), fill='pink')

            # update
            j_curr = j2s[found]
            _curr = j2s_inds[found]

            print(peaks)
            print(found)
            print(edge_scores)
            #print(hist_diff)
            # print(as_j12)
            # print(j2s)
            plt.figure()
            plt.imshow(deb)
            plt.show()
            if found == -1 :
                break
    # print(sorted_juncs)
    #for j1 in junctions:
    return

def tileImages(image_list, n_cols, background_color=0, padding=5):
    image_width = image_list[0][0].shape[1]
    image_height = image_list[0][0].shape[0]
    width = image_width * n_cols + padding * (n_cols + 1)
    height = image_height * len(image_list) + padding * (len(image_list) + 1)
    tiled_image = np.zeros((height, width, 3), dtype=np.uint8)
    tiled_image[:, :] = background_color
    for y, images in enumerate(image_list):
        offset_y = image_height * y + padding * (y + 1)        
        for x, image in enumerate(images):
            offset_x = image_width * x + padding * (x + 1)
            tiled_image[offset_y:offset_y + image_height, offset_x:offset_x + image_width] = image
            continue
        continue
    return tiled_image