import pickle as p
import glob
import svgwrite
import os
import numpy as np
from numpy.random import randn
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from models.graph import EdgeClassifier
from models.resnet import resnet152, resnet18
import torch
import random

def draw_junctions(_id, junctions, path, thetas=None, theta_confs=None):
    # draw corners
    dwg = svgwrite.Drawing(path, (256, 256))
    for i in range(len(junctions)):
        # angs = thetas[i]
        x, y = np.array(junctions[i])/1.0
        if 'PC' in path:
            dwg.add(dwg.circle(center=(x, y), r=3, stroke='green', fill='white', stroke_width=1, opacity=.8))
        else:
            dwg.add(dwg.circle(center=(x, y), r=1, stroke='green', fill='white', stroke_width=1, opacity=.8))
        if thetas is not None:

            for a, c in zip(thetas[i], theta_confs[i]):
                if c > 0.5:
                    rad = np.radians(a)
                    dy = np.sin(rad)*10.0
                    dx = np.cos(rad)*10.0
                    dwg.add(dwg.line((float(x), float(y)), (float(x+dx), float(y+dy)), stroke='red', stroke_width=2, opacity=.8))
    dwg.save()
    return

def draw_edges(_id, edge_map, path):
    
    transp_edge = Image.new('RGBA',  (256, 256), (255, 0, 255, 0))
    transp_edge_arr = np.array(transp_edge)
    transp_edge_arr[:, :, -1] = (edge_map*255.0).astype('uint8')
    transp_edge = Image.fromarray(transp_edge_arr).resize((256, 256))
    transp_edge.save(path.replace('svg', 'png'))
    
    dwg = svgwrite.Drawing(path, (256, 256))
    dwg.add(svgwrite.image.Image(os.path.abspath(path.replace('svg', 'png')), size=(256, 256)))
    dwg.save()
    
    return

def draw_shared_edges(im_path, shared_edges, _id, path):

    bg_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))  # Semitransparent background.
    for i in shared_edges:
        if _id in list(i):
            for m in shared_edges[i]:
                reg = Image.new('RGBA', (256, 256), (0,0,0,0))
                dr_reg = ImageDraw.Draw(reg)
                m[m>0] = 255
                m[m<0] = 0
                m = Image.fromarray(m)
                r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                dr_reg.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 256))
                bg_img.paste(Image.alpha_composite(bg_img, reg))
    bg_img.save(path)
    return

# def draw_regions(regions, _id, path):

#     # draw all regions
#     reg = Image.new('RGBA', (256, 256), (255, 255, 255, 0))
#     dr = ImageDraw.Draw(reg)
#     import matplotlib.pyplot as plt
#     for m in regions:
#         import random
#         r = random.randint(0,255) ; g = random.randint(0,255) ; b = random.randint(0,255)
#         m = Image.fromarray((m*255.0).astype('uint8'))
#         dr.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 128))
#     reg.save(path)
#     return

def draw_regions(masks, _id, path):
    import random, cv2
    bg_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))  # Semitransparent background.
    colors = []
    for m in masks:
        # draw region
        reg = Image.new('RGBA', (256, 256), (0,0,0,0))
        dr_reg = ImageDraw.Draw(reg)
        m[m>0] = 255
        m[m<0] = 0
        m = Image.fromarray(m)
        r = random.randint(0,255) ; g = random.randint(0,255) ; b = random.randint(0,255)
        colors.append((r, g, b))
        dr_reg.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 64))
        bg_img.paste(Image.alpha_composite(bg_img, reg))

    for m, color in zip(masks, colors):
        cnt = Image.new('RGBA', (256, 256), (0,0,0,0))
        dr_cnt = ImageDraw.Draw(cnt)
        mask = np.zeros((256,256,3)).astype('uint8')
        m[m>0] = 255
        m[m<0] = 0
        m = m.astype('uint8')
        m = cv2.resize(m, (256, 256), interpolation = cv2.INTER_AREA) 
        ret,thresh = cv2.threshold(m,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:  
            contours = [c for c in contours]
        r, g, b = color
        cv2.drawContours(mask, contours, -1, (255, 255, 255), 2)
        mask = Image.fromarray(mask)
        dr_cnt.bitmap((0, 0), mask.convert('L'), fill=(r, g, b, 256))
        bg_img.paste(Image.alpha_composite(bg_img, cnt))
    bg_img.save(path)
    return 

def nms(junctions, junc_confs, thetas, theta_confs, nms_thresh=4.0):

    inds = np.argsort(junc_confs)[::-1]
    junc_confs_sorted = np.array(junc_confs[inds])
    juncs_sorted = np.array(junctions[inds])
    thetas_sorted = np.array(thetas[inds])
    theta_confs_sorted = np.array(theta_confs[inds])

    # get loop 
    dists = np.apply_along_axis(np.linalg.norm, 2,\
        juncs_sorted[:, None, :] - juncs_sorted[None, :, :])

    # apply nms
    keep_track = np.zeros(junc_confs_sorted.shape[0])
    nms_inds = []
    for i in range(junc_confs_sorted.shape[0]):
        if (keep_track[i] == 0):
            nms_inds.append(i)
            for j in range(junc_confs_sorted.shape[0]):
                if dists[i, j] < nms_thresh:
                    keep_track[j] = 1

    return juncs_sorted[nms_inds], junc_confs_sorted[nms_inds], thetas_sorted[nms_inds], theta_confs_sorted[nms_inds]

def adjust_learning_rate(optimizer, epoch, LR, LR_param):
    #lr = LR * (0.1 ** (epoch // dropLR))
    LR_policy = LR_param.get('lr_policy', 'step')
    if LR_policy == 'step':
        steppoints = LR_param.get('steppoints', [4, 7, 9, 10])
        lrs = LR_param.get('lrs', [0.001, 0.001, 0.001, 0.001, 0.001])
        assert len(lrs) == len(steppoints) + 1
        
        lr = None
        for idx, steppoint in enumerate(steppoints):
            if epoch > steppoint:
                continue
            elif epoch <= steppoint:
                lr = lrs[idx]
                break
        if lr is None:
            lr = lrs[-1]

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
             
def Rnd(x):
  return max(-2 * x, min(2 * x, randn() * x))
  
def Flip(img):
  return img[:, :, ::-1].copy()  
  
def ShuffleLR(x):
  for e in ref.shuffleRef:
    x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
  return x

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

def filter_angles(th, th_c, thresh=0.5):
    th_filtered = []
    th_c_filtered = []
    for th_list, th_c_list in zip(th, th_c):
        inds = np.where(th_c_list > thresh)
        th_filtered.append(th_list[inds])
        th_c_filtered.append(th_c_list[inds])
    return th_filtered, th_c_filtered

def filter_regions(region_mks, shared_edges, _id, filter_size=11):

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
    inds_sorted = np.array(inds)

    # filter zero sized
    inds = np.where(sizes_sorted>0)
    sizes_sorted = sizes_sorted[inds]
    region_mks_sorted = region_mks_sorted[inds]
    reg_sm_sorted = reg_sm_sorted[inds]
    inds_sorted = inds_sorted[inds]

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
    
    # revert indices
    inds_map = {}
    curr_j = 0
    for k, ind in enumerate(inds_sorted):
        if not suppressed[k]:
            inds_map[ind] = curr_j
            curr_j += 1

    # reindex shared edges
    shared_edges_per_id = {}
    for (_id, old_i, old_j) in shared_edges.keys():
        if (old_i in inds_map) and (old_j in inds_map): 
            shared_edges_per_id[(_id, inds_map[old_i], inds_map[old_j])] = shared_edges[(_id, old_i, old_j)]

    return regions_filtered, shared_edges_per_id

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