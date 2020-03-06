import pickle as p
import glob
import svgwrite
import os
import numpy as np
from utils.optimizer import reconstructBuildingBaseline
from utils.metrics import Metrics
from utils.utils import nms
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from models.graph import EdgeClassifier
from models.resnet import resnet152, resnet18
import torch

def draw_junctions(dwg, junctions, thetas=None, theta_confs=None):

    # draw corners
    for i in range(len(junctions)):

        # angs = thetas[i]
        x, y = np.array(junctions[i])/2.0
        dwg.add(dwg.circle(center=(x, y), r=2, stroke='green', fill='white', stroke_width=1, opacity=.8))

        if thetas is not None:

            for a, c in zip(thetas[i], theta_confs[i]):
                if c > 0.5:
                    rad = np.radians(a)
                    dy = np.sin(rad)*5.0
                    dx = np.cos(rad)*5.0
                    dwg.add(dwg.line((float(x), float(y)), (float(x+dx), float(y+dy)), stroke='red', stroke_width=1, opacity=.8))

    return

def filter_angles(th, th_c, thresh=0.5):
    th_filtered = []
    th_c_filtered = []
    for th_list, th_c_list in zip(th, th_c):
        inds = np.where(th_c_list > thresh)
        th_filtered.append(th_list[inds])
        th_c_filtered.append(th_c_list[inds])
    return th_filtered, th_c_filtered

def filter_regions(region_mks, filter_size=11):

    # sort by size
    sizes = []
    reg_sm = []
    for i, reg_i in enumerate(region_mks):

        # apply min filter 
        reg_i_sm = Image.fromarray(reg_i*255.0)
        reg_i_sm = reg_i_sm.filter(ImageFilter.MinFilter(filter_size))
        reg_i_sm = np.array(reg_i_sm)/255.0
        reg_sm.append(reg_i_sm)
        sizes.append(reg_i_sm.sum())

    inds = np.argsort(sizes)[::-1]
    sizes_sorted = np.array(sizes)[inds]
    region_mks_sorted = np.array(region_mks)[inds]
    reg_sm_sorted = np.array(reg_sm)[inds]

    # filter zero sized
    inds = np.where(sizes_sorted>0)
    sizes_sorted = sizes_sorted[inds]
    region_mks_sorted = region_mks_sorted[inds]
    reg_sm_sorted = reg_sm_sorted[inds]

    # suppress regions
    suppressed = np.zeros(region_mks_sorted.shape[0])
    for i, reg_i in enumerate(region_mks_sorted):
        for j, reg_j in enumerate(region_mks_sorted):
            if i != j and not suppressed[i] and not suppressed[j]:
                intersec = np.logical_and(reg_sm_sorted[i], reg_sm_sorted[j])
                intersec = np.array(np.where(intersec>0))
                if intersec.shape[1] > 0:
                    suppressed[j] = 1

                    # # DEBUG
                    # deb_1 = Image.fromarray(reg_i*255.0)
                    # deb_2 = Image.fromarray(reg_j*255.0)
                    # plt.figure()
                    # plt.imshow(deb_1)
                    # plt.figure()
                    # plt.imshow(deb_2)
                    # plt.show()

    regions_filtered = [m for i, m in enumerate(region_mks_sorted) if not suppressed[i]]
    regions_filtered = np.array(regions_filtered)
    sizes_filtered = [s for i, s in enumerate(sizes_sorted) if not suppressed[i]]
    sizes_filtered = np.array(sizes_filtered)
    print(sizes_filtered)

    return regions_filtered

def load_annots(graph):

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

    return np.array(corner_set), np.array(edge_set)

def draw_building(dwg, junctions, juncs_on, lines_on):

    # draw edges
    for k, l in lines_on:
        x1, y1 = np.array(junctions[k])/2.0
        x2, y2 = np.array(junctions[l])/2.0
        #fill='rgb({},{},{})'.format(*(np.random.rand(3)*255).astype('int'))
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='red', stroke_width=1, opacity=.8))

    # draw corners
    for j in juncs_on:
        x, y = np.array(junctions[j])/2.0
        dwg.add(dwg.circle(center=(x, y), r=2, stroke='green', fill='white', stroke_width=1, opacity=.8))
    return 

def stress(all_numbers):
    num = max([len(numbers) for numbers in all_numbers])
    max_numbers = np.array([numbers for numbers in all_numbers if len(numbers) == num]).max(0)
    for numbers in all_numbers:
        if len(numbers) != num:
            new_line = '\\hline ' + ' & '.join(numbers) + ' \\\\'
            pass
        else:
            numbers = ['%0.1f'%(number) if number != max_numbers[index] else '\textbf{' + '%0.1f'%(number) + '}' for index, number in enumerate(numbers)]
            new_line = '\\hline ' + ' & '.join(numbers) + ' \\\\'
            pass
        print(new_line)
        continue
    return

def get_edge_scores(junctions, regions, rgb_folder, _id, epoch=1, model='resnet152'):

    # load model
    resnet = resnet152(pretrained=False).cuda()
    edge_classifier = EdgeClassifier(resnet)
    edge_classifier = edge_classifier.cuda()
    edge_classifier = edge_classifier.eval()

    # open RGB image
    split = 'det'
    out_size = 256
    rgb_path = os.path.join(rgb_folder, _id +'.jpg')
    rgb = Image.open(rgb_path).resize((out_size, out_size))
    rgb = np.array(rgb)/255.0
    model_path = '/home/nelson/Workspace/building_reconstruction/working_model/binary_edge_classifier_with_regions/saved_models/edge_classifier_{}_{}_iter_{}.pth'.format(model, split, epoch)
    edge_classifier.load_state_dict(torch.load(model_path))

    # check backup -- save time
    temp_dir = './temp/{}/'.format(model)
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)

    backup_path = '{}/{}_{}_{}.npy'.format(temp_dir, _id, epoch, split)
    if os.path.isfile(backup_path):
        return np.load(open(backup_path, 'rb'),  encoding='bytes').item()


    # combine regions
    all_reg = np.zeros((out_size, out_size))
    for k, reg in enumerate(regions):
        reg = Image.fromarray(reg*255.0).resize((out_size, out_size))
        reg = np.array(reg)/255.0
        inds = np.array(np.where(reg==1))
        all_reg[inds[0, :], inds[1, :]] = k

    # draw edge
    lw_from_cls = {}
    for k, c1 in enumerate(junctions):
        for l, c2 in enumerate(junctions):
            if k > l:
                edge = Image.new('L', (out_size, out_size))
                dr = ImageDraw.Draw(edge)
                x1, y1 = c1
                x2, y2 = c2
                div = 256.0/out_size
                dr.line((int(x1/div), int(y1/div), int(x2/div), int(y2/div)), fill="white", width=int(4/div))
                edge = np.array(edge)/255.0
                imgs = np.concatenate([rgb.transpose(2, 0, 1), edge[np.newaxis, :, :], all_reg[np.newaxis, :, :]], 0)
                imgs = torch.from_numpy(imgs).cuda().float()
                prob = edge_classifier(imgs.unsqueeze(0))
                prob = prob.detach().cpu().numpy()
                lw_from_cls[(k, l)] = prob[0]

    # save backup
    np.save(open(backup_path, 'wb'), lw_from_cls)

    return lw_from_cls

res_dir = '/home/nelson/Workspace/outdoor_project/results/junc/3/15/2'
rgb_dir = '/home/nelson/Workspace/cities_dataset/rgb/'
edge_dir = '/home/nelson/Workspace/cities_dataset/edge_map/'
corner_dir = '/home/nelson/Workspace/cities_dataset/dets/corners/'
annot_dir = '/home/nelson/Workspace/cities_dataset/annot/'
region_dir = '/home/nelson/Workspace/cities_dataset/regions/'
metrics = [Metrics(), Metrics(), Metrics(), Metrics(), Metrics(), Metrics(), Metrics()]

with open('/home/nelson/Workspace/cities_dataset/valid_list.txt') as f:
    _ids = [x.strip() for x in f.readlines()]

for _id in _ids:

    # if '1548206121.73' not in _id:
    #     continue
    # # # 1553980237.28

    # load detections
    fname = '{}/{}.jpg_5.pkl'.format(res_dir, _id)
    with open(fname, 'rb') as f:
        c = p.load(f, encoding='latin1')

    # apply non maxima supression
    cs, cs_c, th, th_c = nms(c['junctions'], c['junc_confs'], c['thetas'], c['theta_confs'], nms_thresh=8.0)

    # load annotations
    p_path = '{}/{}.npy'.format(annot_dir, _id)
    v_set = np.load(open(p_path, 'rb'),  encoding='bytes')
    graph_annot = dict(v_set[()])
    cs_annot, es_annot = load_annots(graph_annot)

    # load edge map
    edge_map_path = '{}/{}.jpg'.format(edge_dir, _id)
    im_path = '{}/{}.jpg'.format(rgb_dir, _id)
    edge_map = np.array(Image.open(edge_map_path).convert('L'))/255.0

    # load region masks
    region_path = '{}/{}.npy'.format(region_dir, _id)
    region_mks = np.load(region_path)
    region_mks = filter_regions(region_mks)

    # compute edge scores from classifier
    lw_from_cls = None #get_edge_scores(cs, region_mks, rgb_dir, _id)

    # draw annotations
    dwg = svgwrite.Drawing('../results/svg_regions/{}_7.svg'.format(_id), (128, 128))
    #dwg.add(svgwrite.image.Image(im_path, size=(128, 128)))
    im_path = os.path.join(rgb_dir, _id + '.jpg')
    draw_building(dwg, cs_annot, np.array(range(cs_annot.shape[0])), es_annot)
    dwg.save()

    # filter angles
    th_filtered, th_c_filtered = filter_angles(th, th_c, thresh=0.5)

    ##################### Baseline #######################
    junctions, juncs_on, lines_on = reconstructBuildingBaseline(cs, edge_map,
        use_junctions=True,
        thetas=th_filtered,
        angle_thresh=5,
        intersection_constraint=True,
        post_process=True)

    dwg = svgwrite.Drawing('../results/svg_regions/{}_0.svg'.format(_id), (128, 128))
    #dwg.add(svgwrite.image.Image(im_path, size=(128, 128)))
    im_path = os.path.join(rgb_dir, _id + '.jpg')
    draw_building(dwg, junctions, juncs_on, lines_on)
    dwg.save()
    metrics[0].forward(graph_annot, junctions, juncs_on, lines_on, _id)

    ## EXPERIMENT I - Weighted edges indicator variables using edges confidence ##
    junctions, juncs_on, lines_on = reconstructBuildingBaseline(cs, edge_map,
        use_junctions=True,
        thetas=th_filtered,
        angle_thresh=5,
        edge_threshold=0.5,
        lw_from_cls=lw_from_cls,
        with_edge_confidence=True,
        use_edge_classifier=False,
        edge_map_weight=10.0,
        intersection_constraint=True,
        post_process=True)

    dwg = svgwrite.Drawing('../results/svg_regions/{}_1.svg'.format(_id), (128, 128))
    #dwg.add(svgwrite.image.Image(im_path, size=(128, 128)))
    im_path = os.path.join(rgb_dir, _id + '.jpg')
    draw_building(dwg, junctions, juncs_on, lines_on)
    dwg.save()
    metrics[1].forward(graph_annot, junctions, juncs_on, lines_on, _id)

    ## Experiment II - Weights as combined junctions and edges confidence ##  
    junctions, juncs_on, lines_on = reconstructBuildingBaseline(cs, edge_map,
        use_junctions=True,
        thetas=th_filtered,
        angle_thresh=5,
        with_corner_edge_confidence=True,
        corner_confs=cs_c,
        corner_edge_thresh=0.125,
        lw_from_cls=lw_from_cls,
        use_edge_classifier=False,
        edge_map_weight=10.0,
        intersection_constraint=True,
        post_process=True
        )
    
    dwg = svgwrite.Drawing('../results/svg_regions/{}_2.svg'.format(_id), (128, 128))
    #dwg.add(svgwrite.image.Image(im_path, size=(128, 128)))
    im_path = os.path.join(rgb_dir, _id + '.jpg')
    draw_building(dwg, junctions, juncs_on, lines_on)
    dwg.save()
    metrics[2].forward(graph_annot, junctions, juncs_on, lines_on, _id)

    ## Experiment III - Maximizing weighted sum of junction directions ##    
    junctions, juncs_on, lines_on = reconstructBuildingBaseline(cs, edge_map,
        use_junctions_with_var=True,
        thetas=th,
        angle_thresh=5,
        with_corner_edge_confidence=True,
        corner_confs=cs_c,
        theta_confs=th_c,
        theta_threshold=0.25,
        corner_edge_thresh=0.125,
        lw_from_cls=lw_from_cls,
        use_edge_classifier=False,
        edge_map_weight=10.0,
        intersection_constraint=True,
        post_process=True
        )
    dwg = svgwrite.Drawing('../results/svg_regions/{}_3.svg'.format(_id), (128, 128))
    #dwg.add(svgwrite.image.Image(im_path, size=(128, 128)))
    im_path = os.path.join(rgb_dir, _id + '.jpg')
    draw_building(dwg, junctions, juncs_on, lines_on)
    dwg.save()
    metrics[3].forward(graph_annot, junctions, juncs_on, lines_on, _id)

    ################### EXPERIMENT IV ####################
    junctions, juncs_on, lines_on, regs_sm_on = reconstructBuildingBaseline(cs, edge_map,
        use_junctions_with_var=True,
        use_regions=True,
        thetas=th,
        regions=region_mks,
        angle_thresh=5,
        with_corner_edge_confidence=True,
        corner_confs=cs_c,
        corner_edge_thresh=0.125,
        theta_confs=th_c,
        theta_threshold=0.25,
        region_hit_threshold=0.1,
        lw_from_cls=lw_from_cls,
        use_edge_classifier=False,
        closed_region_lowerbound=True,
        closed_region_upperbound=True,
        with_corner_variables=True,
        corner_min_degree_constraint=True,
        junctions_soft=True,
        edge_map_weight=10.0,
        intersection_constraint=True,
        post_process=True
        )
    im_path = '{}/{}.jpg'.format(rgb_dir, _id)
    deb = Image.fromarray(np.ones((256, 256))*255).convert('RGB') #Image.open(im_path)
    dr = ImageDraw.Draw(deb)
    import matplotlib.pyplot as plt
    for m in regs_sm_on:
        import random
        r = random.randint(0,255) ; g = random.randint(0,255) ; b = random.randint(0,255)
        dr.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 128))
    
    # plt.imshow(deb)
    # plt.show()

    deb.save('./regions/{}.jpg'.format(_id))
    dwg = svgwrite.Drawing('../results/svg_regions/{}_4.svg'.format(_id), (128, 128))
    dwg.add(svgwrite.image.Image(os.path.abspath('./regions/{}.jpg'.format(_id)), size=(128, 128)))
    im_path = os.path.join(rgb_dir, _id + '.jpg')
    draw_building(dwg, junctions, juncs_on, lines_on)
    dwg.save()
    metrics[4].forward(graph_annot, junctions, juncs_on, lines_on, _id)

    ################### EXPERIMENT V ####################
    junctions, juncs_on, lines_on, regs_sm_on = reconstructBuildingBaseline(cs, edge_map,
        use_junctions_with_var=True,
        use_regions=True,
        thetas=th,
        regions=region_mks,
        angle_thresh=5,
        with_corner_edge_confidence=True,
        corner_confs=cs_c,
        corner_edge_thresh=0.125,
        theta_confs=th_c,
        theta_threshold=0.25,
        region_hit_threshold=0.1,
        lw_from_cls=lw_from_cls,
        use_edge_classifier=False,
        closed_region_lowerbound=True,
        closed_region_upperbound=True,
        with_corner_variables=True,
        corner_min_degree_constraint=True,
        junctions_soft=True,
        region_intersection_constraint=True,
        edge_map_weight=10.0,
        intersection_constraint=True,
        post_process=True
        )
    im_path = '{}/{}.jpg'.format(rgb_dir, _id)
    deb = Image.fromarray(np.ones((256, 256))*255).convert('RGB') #Image.open(im_path)
    dr = ImageDraw.Draw(deb)
    import matplotlib.pyplot as plt
    for m in regs_sm_on:
        import random
        r = random.randint(0,255) ; g = random.randint(0,255) ; b = random.randint(0,255)
        dr.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 128))
    
    # plt.imshow(deb)
    # plt.show()

    deb.save('./regions/{}.jpg'.format(_id))
    dwg = svgwrite.Drawing('../results/svg_regions/{}_5.svg'.format(_id), (128, 128))
    dwg.add(svgwrite.image.Image(os.path.abspath('./regions/{}.jpg'.format(_id)), size=(128, 128)))
    im_path = os.path.join(rgb_dir, _id + '.jpg')
    draw_building(dwg, junctions, juncs_on, lines_on)
    dwg.save()
    metrics[5].forward(graph_annot, junctions, juncs_on, lines_on, _id)

   ################### EXPERIMENT VI ####################
    junctions, juncs_on, lines_on, regs_sm_on = reconstructBuildingBaseline(cs, edge_map,
        use_junctions_with_var=True,
        use_regions=True,
        thetas=th,
        regions=region_mks,
        angle_thresh=5,
        with_corner_edge_confidence=True,
        corner_confs=cs_c,
        corner_edge_thresh=0.125,
        theta_confs=th_c,
        theta_threshold=0.25,
        region_hit_threshold=0.1,
        lw_from_cls=lw_from_cls,
        use_edge_classifier=False,
        closed_region_lowerbound=True,
        closed_region_upperbound=True,
        with_corner_variables=True,
        corner_min_degree_constraint=True,
        junctions_soft=True,
        region_intersection_constraint=True,
        inter_region_constraint=True,
        edge_map_weight=10.0,
        inter_region_weight=1.0,
        intersection_constraint=True,
        post_process=True
        )

    # DEBUG
    im_path = '{}/{}.jpg'.format(rgb_dir, _id)
    deb = Image.fromarray(np.ones((256, 256))*255).convert('RGB') # Image.open(im_path)
    dr = ImageDraw.Draw(deb)
    import matplotlib.pyplot as plt
    for m in regs_sm_on:
        import random
        r = random.randint(0,255) ; g = random.randint(0,255) ; b = random.randint(0,255)
        dr.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 128))

    deb.save('./regions/{}.jpg'.format(_id))
    dwg = svgwrite.Drawing('../results/svg_regions/{}_6.svg'.format(_id), (128, 128))
    dwg.add(svgwrite.image.Image(os.path.abspath('./regions/{}.jpg'.format(_id)), size=(128, 128)))
    im_path = os.path.join(rgb_dir, _id + '.jpg')
    draw_building(dwg, junctions, juncs_on, lines_on)
    dwg.save()
    metrics[6].forward(graph_annot, junctions, juncs_on, lines_on, _id)

# print metrics
all_results = []
for k, m in enumerate(metrics):
    print('experiment %d'%(k))
    values =  m.print_metrics()
    values = [x*100.0 for x in values]
    all_results.append(values)
stress(all_results)