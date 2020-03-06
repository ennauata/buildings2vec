from gurobipy import *
import cv2
import numpy as np
import sys
import csv
import copy
from utils import *
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import matplotlib.pyplot as plt
from utils.intersections import doIntersect
from skimage import measure
from rdp import rdp

def reconstructBuildingBaseline(junctions, edge_map, regions=None, with_weighted_junctions=True, with_corner_variables=False, with_edge_confidence=False, 
    with_corner_edge_confidence=False, \
    lw_from_cls=None, use_edge_classifier=False,\
    corner_min_degree_constraint=False, ignore_invalid_corners=False, use_junctions_with_var=False, \
    use_regions=False, corner_suppression=False, corner_penalty=False, \
    junction_suppression=False, \
    intersection_constraint=False, angle_constraint=False, use_junctions=False, use_loops=False, \
    dist_thresh=None, angle_thresh=None, edge_threshold=None, corner_edge_thresh=None, thetas=None, corner_confs=None, \
    theta_threshold=0.2, region_hit_threshold=None, theta_confs=None, filter_size=11, \
    region_weight=1000.0, post_process=False, \
    edge_map_weight=1.0, junctions_weight=1.0, inter_region_weight=1.0, wrong_dir_weight=1.0, closed_region_weight=1.0,
    closed_region_lowerbound=False, closed_region_upperbound=False,\
    region_intersection_constraint=False, inter_region_constraint=False,\
    junctions_soft=False, shared_edges=None, _id=None, _exp_tag=''):

    # create a new model
    m = Model("building_reconstruction_baseline")
    m.setParam( 'OutputFlag', False )
    obj = LinExpr(0)
    num_junc = len(junctions)

    # list primitives
    js_list = [k for k in range(num_junc)]

    if ignore_invalid_corners:
        js_list = [k for k in js_list if len(thetas[k]) >= 2]

    ls_list = [(k, l) for k in js_list for l in js_list if l > k]

    # create variables
    if with_corner_variables:
        js_var_dict = {}
        for j in js_list:
            js_var_dict[j] = m.addVar(vtype = GRB.BINARY, name="junc_{}".format(j))

    ls_var_dict = {}
    for k, l in ls_list:
        ls_var_dict[(k, l)] = m.addVar(vtype = GRB.BINARY, name="line_{}_{}".format(k, l))

    # edgeness objective
    if with_edge_confidence:
        for k, l in ls_list:
            if use_edge_classifier:
                try:
                    lw = lw_from_cls[(k, l)]
                except:
                    lw = lw_from_cls[(l, k)]
            else:
                lw = getLineWeight(edge_map, junctions[k], junctions[l]) 
            obj += (lw-edge_threshold)*ls_var_dict[(k, l)] # favor edges with over .5?

    elif with_corner_edge_confidence:
        for k, l in ls_list:
            if use_edge_classifier:
                try:
                    lw = lw_from_cls[(k, l)]
                except:
                    lw = lw_from_cls[(l, k)]
            else:
                lw = getLineWeight(edge_map, junctions[k], junctions[l]) 
            #obj += (lw-0.1)*ls_var_dict[(k, l)]
            #obj += (corner_confs[k]-corner_threshold)*(corner_confs[l]-corner_threshold)*(lw-edge_threshold)*ls_var_dict[(k, l)] # favor edges with over .5?
            #print((np.prod([corner_confs[k], corner_confs[l], lw])-corner_edge_thresh))
            obj += edge_map_weight*(np.prod([corner_confs[k], corner_confs[l], lw])-corner_edge_thresh)*ls_var_dict[(k, l)] # favor edges with over .5?
    else:
        for k, l in ls_list:
            obj += ls_var_dict[(k, l)]

    if with_corner_variables:
        # corner-edge connectivity constraint
        for k, l in ls_list:
            m.addConstr((js_var_dict[k] + js_var_dict[l] - 2)*ls_var_dict[(k, l)] == 0, "c_{}_{}".format(k, l))

##########################################################################################################
############################################### OPTIONAL #################################################
##########################################################################################################
    if use_regions:
        reg_list = []
        reg_var_ls = {}
        reg_sm ={}
        reg_contour = {} 
        for i, reg in enumerate(regions):
            
            # apply min filter 
            reg_small = Image.fromarray(reg*255.0)
            reg_small = reg_small.filter(ImageFilter.MinFilter(filter_size))
            reg_small = np.array(reg_small)/255.0

            # ignore too small regions
            inds = np.argwhere(reg_small>0)
            if np.array(inds).shape[0] > 0:

                ret, thresh = cv2.threshold(np.array(reg_small*255.0).astype('uint8'), 127, 255, 0)
                _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                lg_contour = [None, 0]

                for c in contours:
                    cont = Image.new('L', (256, 256))
                    dr = ImageDraw.Draw(cont)
                    c = c.reshape(-1, 2)
                    c = [(x, y) for x, y in c]
                    if len(c) <= 2:
                        continue
                    dr.polygon(c, fill='white')
                    size = (np.array(cont)/255.0).sum()
                    if size > lg_contour[1]:
                        lg_contour = [c, size]
                contours = lg_contour[0]
                contours = np.array(contours)

                if len(contours.shape) > 0:
                    reg_contour[i] = contours.reshape(-1, 2)
                    inds = np.linspace(0, reg_contour[i].shape[0], min(int(reg_contour[i].shape[0]/2), reg_contour[i].shape[0]), endpoint=False).astype('int')
                    reg_contour[i] = reg_contour[i][inds, :]
                    reg_list.append(i)
                    reg_var_ls[i] = m.addVar(vtype = GRB.BINARY, name="reg_{}".format(i))
                    reg_sm[i] = reg_small
                    obj += region_weight*reg_var_ls[i]

        for i in reg_list:
            if region_intersection_constraint:
                # compute intersection constraint
                for k, l in ls_list:
                    intersec = getIntersection(reg_sm[i], junctions[k], junctions[l])
                    if intersec >= region_hit_threshold:
                        m.addConstr(ls_var_dict[(k, l)]*reg_var_ls[i] == 0, "r1_{}_{}".format(k, l))

            
            # add loop constraints
            ths, pts = compute_normals(reg_contour[i], reg_sm[i])
            other_regions = [regions[j] for j in reg_list if i != j]
            sm_other_regions = [reg_sm[j] for j in reg_list if i != j]
            for pt, th in zip(pts, ths):

                # closed region soft constraint -- upperbound
                intersec_edges, intersec_region, self_intersect = castRay(pt, th, ls_list, junctions, regions[i], reg_sm[i], other_regions, ray_length=1000.0)
                sum_in_set = LinExpr(0)
                for e in list(intersec_edges):
                    k, l = e
                    sum_in_set += ls_var_dict[(k, l)]
                if not intersec_region and closed_region_upperbound:
                    slack_var = m.addVar(vtype=GRB.INTEGER, name="slack_var_{}_{}".format(th, i))
                    m.addConstr(sum_in_set*reg_var_ls[i] <= reg_var_ls[i] + slack_var, "r2_{}_{}".format(th, i))
                    obj -= closed_region_weight * slack_var
                    m.addConstr(slack_var >= 0)

                if closed_region_lowerbound:
                    # closed region hard constraint -- lowerbound
                    slack_var = m.addVar(vtype=GRB.INTEGER)
                    m.addConstr(sum_in_set*reg_var_ls[i] >= reg_var_ls[i]-slack_var, "r2_{}".format(th))
                    obj -= closed_region_weight * slack_var
                    m.addConstr(slack_var >= 0)

        ################ WORKING HERE ##########################
        # compute shared edge constraint
        if shared_edges is not None:
            for i in reg_list:
                for j in reg_list:
                    if (_id, i, j) in shared_edges:
                        for k, edge_msk in enumerate(shared_edges[(_id, i, j)]):

                            ## DEBUG ##
                            deb_im = Image.new('RGB', (256, 256))
                            deb_im_arr = np.array(deb_im)

                            inds = np.array(np.where(reg_sm[i] > 0))
                            deb_im_arr[inds[0, :], inds[1, :], 0] = 255

                            inds = np.array(np.where(reg_sm[j] > 0))
                            deb_im_arr[inds[0, :], inds[1, :], 1] = 255
                            
                            inds = np.array(np.where(edge_msk > 0))
                            deb_im_arr[inds[0, :], inds[1, :], 2] = 255

                            deb_im = Image.fromarray(deb_im_arr.astype('uint8'))
                            dr = ImageDraw.Draw(deb_im)
                            ## DEBUG ##
                            n_inter = 0
                            sum_in_set = LinExpr(0)
                            for j1, j2 in ls_list: 
                                # sssprint(getIntersection(edge_msk, junctions[j1], junctions[j2]))
                                if getIntersection(edge_msk, junctions[j1], junctions[j2]) > 0.2:
                                    sum_in_set += ls_var_dict[(j1, j2)]
                                    x1, y1 = junctions[j1]
                                    x2, y2 = junctions[j2]
                                    n_inter += 1
                                    dr.line((x1, y1, x2, y2), fill='magenta', width=1)

                            if n_inter > 0:
                                slack_var_up = m.addVar(vtype=GRB.INTEGER, name="slack_var_inter_up_{}_{}_{}".format(_id, i, j))
                                slack_var_low = m.addVar(vtype=GRB.INTEGER, name="slack_var_inter_low_{}_{}_{}".format(_id, i, j))
                                m.addConstr(sum_in_set >= 1 - slack_var_low)
                                m.addConstr(sum_in_set <= 1 + slack_var_up)
                                m.addConstr(slack_var_low >= 0)
                                m.addConstr(slack_var_up >= 0)
                                obj -= inter_region_weight * slack_var_low + inter_region_weight * slack_var_up

                            ## DEBUG ##
                            deb_im.save('/home/nelson/Workspace/outdoor_project/results/dump/{}_{}_{}_{}.jpg'.format(_id, i, j, k))
                            ## DEBUG ##

    if intersection_constraint:
        # intersection constraint
        for k, (j0, j1) in enumerate(ls_list):
            for l, (j2, j3) in enumerate(ls_list):
                if l > k:
                    p1, q1 = junctions[j0], junctions[j1]
                    p2, q2 = junctions[j2], junctions[j3]
                    if doIntersect(p1, q1, p2, q2):
                        m.addConstr(ls_var_dict[(j0, j1)]*ls_var_dict[(j2, j3)] == 0, "i_{}_{}_{}_{}".format(j0, j1, j2, j3))

    if use_junctions_with_var or use_junctions:

        for j1 in js_list:

            # consider only valid degrees
            # if len(thetas[j1]) >= 2:

            # create list of lines for each junction
            lines_sets = [LinExpr(0) for _ in range(len(thetas[j1])+1)]
            lines_sets_deb = [list() for _ in range(len(thetas[j1])+1)]
            lines_max_in_sets = [0.0 for _ in range(len(thetas[j1])+1)]
            for j2 in js_list:
                if j1 != j2:
                    
                    # get line var
                    if (j1, j2) in ls_var_dict:
                        ls_var = ls_var_dict[(j1, j2)]  
                    else:
                        ls_var = ls_var_dict[(j2, j1)]

                    # check each line angle at junction
                    in_sets = False
                    for i, a in enumerate(thetas[j1]):

                        lb = (a-angle_thresh) if (a-angle_thresh) >= 0 else 360.0+(a-angle_thresh)
                        up = (a+angle_thresh)%360.0

                        pt1 = junctions[j1]
                        pt2 = junctions[j2]
                        ajl = getAngle(pt1, pt2)
                        if inBetween(ajl, lb, up):
                            if use_edge_classifier:
                                try:
                                    lw = lw_from_cls[(j1, j2)]
                                except:
                                    lw = lw_from_cls[(j2, j1)]
                            else:
                                lw = getLineWeight(edge_map, pt1, pt2) 
                            lines_max_in_sets[i] = max(lines_max_in_sets[i], lw)
                            lines_sets[i] += ls_var
                            in_sets = True
                        #print(i, j1, j2, ajl, a, lb, up, inBetween(ajl, lb, up))

                    # not in any direction set 
                    if not in_sets:
                        lines_sets[-1] += ls_var

            # print(lines_sets_deb)
            # # Debug   
            # x1, y1 = junctions[j1]
            # for angle_i, line_set in enumerate(lines_sets_deb):
            #   im_deb = Image.new('RGB', (256, 256))
            #   dr = ImageDraw.Draw(im_deb)
            #   dr.ellipse((x1-2, y1-2, x1+2, y1+2), fill='blue')
            #   for v in line_set:
            #       x2, y2 = junctions[v]
            #       dr.line((x1, y1, x2, y2), fill='green', width=2)
            #       dr.ellipse((x2-2, y2-2, x2+2, y2+2), fill='red')
            #   if angle_i < len(thetas[j1]):
            #       print(thetas[j1][angle_i])
            #   else:
            #       print('Others')
            #   plt.imshow(im_deb)
            #   plt.show()

            # add to constraints
            #set_sum = QuadExpr(0)

            # add all sets
            for i in range(len(thetas[j1])):

                if use_junctions_with_var:
                    junc_th_var = m.addVar(vtype = GRB.BINARY, name="angle_{}".format(j1))
                    obj += junctions_weight*junc_th_var*(np.prod([corner_confs[j1], theta_confs[j1][i]]) - theta_threshold)
                    m.addConstr(lines_sets[i] == junc_th_var, "a_{}_{}".format(i, j1))
                    #  OLD
                    #obj += (np.prod([lines_max_in_sets[i], theta_confs[j1][i]])-theta_threshold)*junc_th_var
                    #set_sum += junc_th_var*lines_sets[i]
                    #m.addConstr(lines_sets[i] <= 1.0, "a_{}_{}".format(i, j1))
                else:
                    m.addConstr(lines_sets[i] <= 1.0, "a_{}_{}".format(i, j1))
                    #  OLD
                    #set_sum += junc_th_var*lines_sets[i]

            # # add not in set -- SOFT
            if junctions_soft:
                slack_var = m.addVar(vtype=GRB.INTEGER, name="slack_var_{}".format(j1))
                m.addConstr(lines_sets[-1] - slack_var == 0, "a_{}_{}".format(-1, j1))
                obj -= wrong_dir_weight * slack_var
                m.addConstr(slack_var >= 0)
            else:
                # add not in set -- HARD
                m.addConstr(lines_sets[-1] == 0, "a_{}_{}".format(-1, j1))

            #  OLD
            #set_sum += junc_th_var*lines_sets[-1]
            # if use_junctions_with_var:
            #   # final constraint
            #   m.addConstr(set_sum == junc_th_var*len(thetas[j1]), "a_sum_{}".format(j1))


    if corner_suppression:

        # junction spatial constraint
        junc_sets = set()
        for j1 in js_list:
            junc_intersec_set = []
            for j2 in js_list:
                pt1 = np.array(junctions[j1])
                pt2 = np.array(junctions[j2])
                dist = np.linalg.norm(pt1-pt2)
                if dist < dist_thresh:
                    junc_intersec_set.append(j2)
            junc_intersec_set = tuple(np.sort(junc_intersec_set))
            junc_sets.add(junc_intersec_set)

        # avoid duplicated constraints
        for js_tuple in junc_sets:
            junc_expr = LinExpr(0)
            for j in np.array(js_tuple):
                junc_expr += js_var_dict[j]
            m.addConstr(junc_expr <= 1, "s_{}".format(j1))

    if junction_suppression:
        angs = []
        for j1 in js_list:

            # compute angles and degree at each junction
            deg_j1 = QuadExpr(0)
            angs = []
            for j2 in js_list:

                if j1 != j2:
                    deg_j1 += ls_var_dict[(j1, j2)] if (j1, j2) in ls_var_dict else ls_var_dict[(j2, j1)]
                    pt1 = junctions[j1]
                    pt2 = junctions[j2]
                    a12 = getAngle(pt1, pt2)
                    angs.append(a12)
                else:
                    angs.append(0.0)

            angs = np.array(angs)
            ang_diffs = np.abs(180.0 - np.abs(angs[:, np.newaxis]-angs[np.newaxis, :]))
            inds = np.array(np.where(ang_diffs <= 10.0)).transpose(1, 0)
            keep_track = []
            for j2, j3 in inds:
                if j1 != j2 and j2 != j3 and j3 != j1:
                    js = tuple(np.sort([j1, j2, j3]))
                    if js not in keep_track:
                        keep_track.append(js)
                        ls_var_12 = ls_var_dict[(j1, j2)] if (j1, j2) in ls_var_dict else ls_var_dict[(j2, j1)]
                        ls_var_13 = ls_var_dict[(j1, j3)] if (j1, j3) in ls_var_dict else ls_var_dict[(j3, j1)]
                        m.addConstr((deg_j1-2)*js_var_dict[j1] >= ls_var_12*ls_var_13 , "j_3_{}_{}_{}".format(j1, j2, j3))  

    if corner_min_degree_constraint:
        # degree constraint
        for j in js_list:

            # degree expression
            deg_j = QuadExpr(0)
            for k, l in ls_list:
                if (j == k) or (j == l):
                    deg_j += ls_var_dict[(k, l)]

            # degree constraint - active junctions must have degree >= 2
            m.addConstr(deg_j*js_var_dict[j] >= 2*js_var_dict[j], "d_1_{}".format(j))

    # set optimizer
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    # parse solution
    juncs_on = []
    lines_on = []
    regs_sm_on = []
    print(_id)
    for v in m.getVars():
        if 'junc' in v.varName and v.x >= .5:
            juncs_on.append(int(v.varName.split('_')[-1]))
        elif 'line' in v.varName and v.x >= .5:
            lines_on.append((int(v.varName.split('_')[-2]), int(v.varName.split('_')[-1])))
        elif 'reg' in v.varName and v.x >= .5:
            #print('REGION ON')
            reg_id = int(v.varName.split('_')[-1])
            reg = regions[reg_id]
            reg_small = Image.fromarray(reg*255.0)
            reg_small = reg_small.filter(ImageFilter.MinFilter(filter_size))
            regs_sm_on.append(reg_small)
        elif 'slack_var_inter' in v.varName:
            continue
            # print(v.varName, v.x)

    if not with_corner_variables:
        juncs_on = np.array(list(set(sum(lines_on, ()))))

    if use_regions:
        if post_process:
            juncs_on, lines_on = remove_junctions(junctions, juncs_on, lines_on)
        return junctions, juncs_on, lines_on, regs_sm_on

    if post_process:
        juncs_on, lines_on = remove_junctions(junctions, juncs_on, lines_on)

    return junctions, juncs_on, lines_on

##########################################################################################################
############################################ HELPER FUNCTIONS ############################################
##########################################################################################################

def remove_junctions(junctions, juncs_on, lines_on, delta=10.0):

    curr_juncs_on, curr_lines_on = list(juncs_on), list(lines_on)
    while True:
        new_lines_on, new_juncs_on = [], []
        is_mod = False
        for j1 in curr_juncs_on:
            adj_js, adj_as, ls = [], [], []
            for j2 in curr_juncs_on:
                if ((j1, j2) in curr_lines_on) or ((j2, j1) in curr_lines_on):
                    adj_js.append(j2)
                    pt1 = junctions[j1]
                    pt2 = junctions[j2]
                    adj_as.append(getAngle(pt1, pt2))
                    ls.append((j1, j2))

            if len(adj_js) > 2 or is_mod or len(adj_js) == 1:
                new_juncs_on.append(j1)
                new_lines_on += ls
            elif len(adj_js) == 2:
                diff = np.abs(180.0-np.abs(adj_as[0]-adj_as[1]))
                if diff >= delta:
                    new_juncs_on.append(j1)
                    new_lines_on += ls
                else:
                    new_lines_on.append((adj_js[0], adj_js[1]))
                    is_mod = True
        curr_juncs_on, curr_lines_on = list(new_juncs_on), list(new_lines_on)
        if not is_mod:
            break

    return curr_juncs_on, curr_lines_on

def compute_normals(contour, reg_sm):

    # deb = Image.fromarray(reg_sm*255.0).convert('RGB')
    # dr = ImageDraw.Draw(deb)
    thetas = []
    points = []
    p1 = contour[-1]
    for k, p2 in enumerate(contour):

        # compute ray
        p3 = contour[(k+1)%contour.shape[0]]
        a21 = getAngle(p2, p1)
        a23 = getAngle(p2, p3)
        an = (a21+a23)/2.0
        ray_im = Image.new('L', (256, 256))
        draw = ImageDraw.Draw(ray_im)
        x1, y1 = p2
        rad = np.radians(an)
        dy = np.sin(rad)*10.0
        dx = np.cos(rad)*10.0
        x2, y2 = x1+dx, y1+dy
        draw.line((x1, y1, x2, y2), fill='white')
        ray = np.array(ray_im)/255.0
        intersect = np.logical_and(ray, reg_sm).sum()/(ray.sum()+1e-8)
        if intersect > 0.2:
            an = (an+180)%360
        thetas.append(an)
        points.append(p2)

        # # debug
        # x1, y1 = p2
        # rad = np.radians(an)
        # dy = np.sin(rad)*10.0
        # dx = np.cos(rad)*10.0
        # x2, y2 = x1+dx, y1+dy
        
        # x3, y3 = p3
        # x4, y4 = p1
        # print(intersect)
        # # dr.ellipse((x4-1, y4-1, x4+1, y4+1), fill='blue')
        # # dr.ellipse((x3-1, y3-1, x3+1, y3+1), fill='blue')
        # dr.ellipse((x1-1, y1-1, x1+1, y1+1), fill='red')
        # dr.line((x1, y1, x2, y2), fill='green')
        # # debug

        p1 = np.array(p2)

    # plt.imshow(deb)
    # plt.show()
    return thetas, points

def castRayRegion(pt, th, ls_list, junctions, sm_reg_i, sm_other_regs, l_reg, l_other_regs, other_regs_id, ray_length=1000.0, thresh=0.0):

    # compute ray
    x1, y1 = int(pt[0]), int(pt[1])
    rad = np.radians(th)
    dy = np.sin(rad)*ray_length
    dx = np.cos(rad)*ray_length
    x2, y2 = x1+dx, y1+dy
    p1, q1 = (x1, y1), (x2, y2)

    # render ray
    ray_im = Image.new('L', (256, 256))
    draw = ImageDraw.Draw(ray_im)
    draw.line((x1, y1, x2, y2), fill='white')
    ray = np.array(ray_im)

    # check region intersection
    background = np.zeros((256, 256))
    comb_reg = np.zeros((256, 256))
    for _id, l_reg_j, sm_reg_j in zip(other_regs_id, l_other_regs, sm_other_regs):
        reg_j_tag = np.array(sm_reg_j)
        inds = np.where(reg_j_tag>0)
        reg_j_tag[inds] = _id+1
        comb_reg += reg_j_tag
        background += l_reg_j
    background += l_reg

    intersect = np.logical_and(ray, comb_reg)
    inds = np.where(intersect>0)
    intersect_tags = np.array(comb_reg[inds])

    # DEBUG
    comb_reg = np.clip(comb_reg, 0, 1)
    comb_reg = Image.fromarray(comb_reg*255.0).convert('RGB')
    dr = ImageDraw.Draw(comb_reg)
    dr.line((x1, y1, x2, y2), fill='green')
    dr.ellipse((x1-2, y1-2, x1+2, y1+2), fill='blue')
    # DEBUG

    # compute distances
    intersec_set = set()
    self_intersect = None
    region_id = None
    inds = np.array(inds)
    if inds.shape[1] > 0:
        px = np.array([x1, y1])[np.newaxis, :]
        pts = inds.transpose(1, 0)[:, ::-1]
        dists = np.sqrt((np.sum((pts-px)**2, -1)))
        closest_pt = pts[np.argmin(dists), :]
        region_id = intersect_tags[np.argmin(dists)]-1
        xc, yc = closest_pt

        # collect intersecting edges
        for i, ls in enumerate(ls_list):
            k, l = ls
            p2, q2 = junctions[k], junctions[l]
            if doIntersect(p1, closest_pt, p2, q2):
                intersec_set.add((k, l))

        # DEBUG -- Draw intersecting edges
        for ls in list(intersec_set):
            k, l = ls
            p2, q2 = junctions[k], junctions[l]
            x3, y3 = p2
            x4, y4 = q2
            dr.line((x3, y3, x4, y4), fill='red', width=1)
        # DEBUG

        # DEBUG
        dr.ellipse((xc-2, yc-2, xc+2, yc+2), fill='magenta')
        #print(region_id)
        # DEBUG

        # check self intersection
        ray_im = Image.new('L', (256, 256))
        dr = ImageDraw.Draw(ray_im) 
        dr.line((x1, y1, xc, yc), fill='white', width=16)
        dr.ellipse((x1-4, y1-4, x1+4, y1+4), fill='black')
        ray = np.array(ray_im)/255.0
        self_intersect = np.logical_and(ray, sm_reg_i).sum()/(ray.sum()+1e-8)
        self_intersect = self_intersect > thresh

        # render ray
        ray_im = Image.new('L', (256, 256))
        draw = ImageDraw.Draw(ray_im)
        draw.line((x1, y1, xc, yc), fill='white', width=16)
        ray = np.array(ray_im)/255.0

        # check background intersection
        background_im = Image.fromarray(background*255.0)
        background_im = background_im.filter(ImageFilter.MaxFilter(9))
        background = 1.0-(np.array(background_im)/255.0)
        background[background>0.5] = 1.0
        background[background<=0.5] = 0.0
        background_intersection = np.logical_and(ray, background).sum()
        if background_intersection > 0:
            self_intersect = True

        # # DEBUG
        # deb = Image.fromarray(background*255.0).convert('RGB')
        # dr = ImageDraw.Draw(deb)
        # dr.line((x1, y1, xc, yc), fill='green')
        # print(background_intersection)
        # plt.imshow(deb)
        # plt.show()

        # # DEBUG
        # print(self_intersect)
        # plt.imshow(comb_reg)
        # plt.show()

    return intersec_set, region_id, self_intersect

def castRayBetweenRegions(p1, q1, ls_list, junctions, reg_i, reg_j):

    # collect intersecting edges
    intersec_set = set()
    for i, ls in enumerate(ls_list):
        k, l = ls
        p2, q2 = junctions[k], junctions[l]
        if doIntersect(p1, q1, p2, q2):
            intersec_set.add((k, l))
            # print(k, l)
            # print(intersec_set)            
            # # DEBUG
            # comb_reg = np.stack([reg_i, reg_j])
            # comb_reg = np.clip(np.sum(comb_reg, 0), 0, 1)
            # comb_reg = Image.fromarray(comb_reg*255.0).convert('RGB')
            # dr = ImageDraw.Draw(comb_reg)
            # x1, y1 = p1
            # x2, y2 = q1
            # x3, y3 = p2
            # x4, y4 = q2
            # dr.line((x1, y1, x2, y2), fill='green', width=4) 
            # dr.line((x3, y3, x4, y4), fill='red', width=1) 
            # print(doIntersect(p1, q1, p2, q2))
            # plt.figure()
            # plt.imshow(comb_reg)
            # plt.show()
        
    return intersec_set

def castRay(pt, th, ls_list, junctions, large_region, region_small, other_regions, ray_length=1000.0, thresh=0.0):

    # compute ray
    x1, y1 = int(pt[0]), int(pt[1])
    rad = np.radians(th)
    dy = np.sin(rad)*ray_length
    dx = np.cos(rad)*ray_length
    x2, y2 = x1+dx, y1+dy

    # collect intersecting edges
    intersec_set = set()
    for i, ls in enumerate(ls_list):
        k, l = ls
        p1, q1 = (x1, y1), (x2, y2)
        p2, q2 = junctions[k], junctions[l]
        if doIntersect(p1, q1, p2, q2):
            intersec_set.add((k, l))

    # # DEBUG
    # deb = Image.fromarray(region_small*255.0).convert('RGB')
    # dr = ImageDraw.Draw(deb) 
    # for ls in list(intersec_set):
    #     k, l = ls
    #     p2, q2 = junctions[k], junctions[l]
    #     x3, y3 = p2
    #     x4, y4 = q2
    #     dr.line((x3, y3, x4, y4), fill='red', width=1)
    # dr.line((x1, y1, x2, y2), fill='green', width=4)
    # plt.imshow(deb)
    # plt.show()
    

    # check intersection with other regions
    intersec_region = False
    if len(other_regions) > 0:
        comb_reg = np.clip(np.sum(np.array(other_regions), 0), 0, 1)

        # cast ray
        ray_im = Image.new('L', (256, 256))
        dr = ImageDraw.Draw(ray_im) 
        dr.line((x1, y1, x2, y2), fill='white', width=8)
        ray = np.array(ray_im)/255.0
        intersec = np.array(np.where(np.logical_and(ray, comb_reg)>0))
        intersec_region = (intersec.shape[1] > 0)
        
        # # DEBUG
        # print(comb_reg.shape)
        # comb_reg = Image.fromarray(comb_reg*255.0).convert('RGB')
        # dr = ImageDraw.Draw(comb_reg) 
        # dr.line((x1, y1, x2, y2), fill='green', width=4)

        # print(intersec)
        # print(intersec_region)
        # plt.figure()
        # plt.imshow(comb_reg)
        # plt.show()
    
    # check self intersection
    ray_im = Image.new('L', (256, 256))
    dr = ImageDraw.Draw(ray_im) 
    dr.line((x1, y1, x2, y2), fill='white', width=8)
    dr.ellipse((x1-4, y1-4, x1+4, y1+4), fill='black')
    ray = np.array(ray_im)/255.0
    self_intersect = np.logical_and(ray, region_small).sum()/(ray.sum()+1e-8)
    self_inter = False
    if self_intersect > thresh:
        intersec_region = True
        self_inter = True
    # # DEBUG
    # print(self_intersect)
    # deb = Image.fromarray(region_small*255.0).convert('RGB')
    # dr = ImageDraw.Draw(deb) 
    # # for k, l in list(intersec_set):
    # #     p2, q2 = junctions[k], junctions[l]
    # #     x3, y3 = p2
    # #     x4, y4 = q2
    # #     dr.line((x3, y3, x4, y4), fill='red', width=1)

    # print(intersec_region)

    # dr.line((x1, y1, x2, y2), fill='green', width=1)
    # plt.imshow(deb)
    # plt.show()
    # # DEBUG

    return intersec_set, intersec_region, self_inter

def getIntersection(region_map, j1, j2):
    x1, y1 = j1
    x2, y2 = j2
    m = Image.new('L', (256, 256))
    dr = ImageDraw.Draw(m)
    dr.line((x1, y1, x2, y2), width=1, fill='white')
    m = np.array(m)/255.0
    inds = np.array(np.where(np.array(m) > 0.0))

    # # DEBUG
    # deb = Image.fromarray(region_map*255.0).convert('RGB')
    # dr = ImageDraw.Draw(deb) 
    # dr.line((x1, y1, x2, y2), fill='red', width=1)
    # print(np.logical_and(region_map, m).sum()/inds.shape[1])
    # plt.imshow(deb)
    # plt.show()
    return np.logical_and(region_map, m).sum()/inds.shape[1]

def getLineWeight(edge_map, j1, j2):

    x1, y1 = j1
    x2, y2 = j2
    m = Image.new('L', (256, 256))
    dr = ImageDraw.Draw(m)
    dr.line((x1, y1, x2, y2), width=2, fill='white')
    inds = np.array(np.where(np.array(m) > 0.0))
    weight = np.sum(edge_map[inds[0, :], inds[1, :]])/inds.shape[1]
    return weight

def getDistanceWeight(region_map, j1):
    x1, y1 = j1
    #x2, y2 = j2
    #line_center = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
    inds = np.argwhere(region_map==1)
    region_center = (inds.sum(0)/inds.shape[0])[::-1]
    d1 = np.linalg.norm(j1-region_center)/np.linalg.norm([255, 255])
    # d2 = np.linalg.norm(j2-region_center)/np.linalg.norm([255, 255])
    #d3 = np.linalg.norm(line_center-region_center)/np.linalg.norm([255, 255])
    return d1

def getAngle(pt1, pt2):
    # return angle in clockwise direction
    x, y = pt1
    xn, yn = pt2
    dx, dy = xn-x, yn-y
    dir_x, dir_y = (dx, dy)/(np.linalg.norm([dx, dy])+1e-8)
    rad = np.arctan2(-dir_y, dir_x)
    ang = np.degrees(rad)
    if ang < 0:
        ang = (ang + 360) % 360
    return 360-ang

def inBetween(n, a, b):
    n = (360 + (n % 360)) % 360
    a = (3600000 + a) % 360
    b = (3600000 + b) % 360
    if (a < b):
        return a <= n and n <= b
    return a <= n or n <= b

def filterOutlineEdges(ls_list, junctions, angles, angle_thresh):

    # filter edges using angles
    new_ls_list = []
    for l in ls_list:
        j1, j2 = l
        pt1 = junctions[j1]
        pt2 = junctions[j2]
        a12 = getAngle(pt1, pt2)
        a21 = getAngle(pt2, pt1)
        drop_edge_at_1 = True
        drop_edge_at_2 = True
        for a1 in angles[j1]:
            if inBetween(a1, a12-angle_thresh, a12+angle_thresh):
                drop_edge_at_1 = False
        for a2 in angles[j2]:
            if inBetween(a2, a21-angle_thresh, a21+angle_thresh):
                drop_edge_at_2 = False
        if (not drop_edge_at_1) and (not drop_edge_at_2):
            new_ls_list.append(l)

    return new_ls_list
if __name__ == '__main__':
    print(inBetween(355, 350, 10))
    print(inBetween(10, 0, 10))
    print(inBetween(0, 0, 10))
    print(inBetween(50, 20, 30))
    print(inBetween(20, 310, 30))