Vectorizing World Buildings
======

Code and instructions for our paper:
[Vectorizing World Buildings: Planar Graph Reconstruction by Primitive Detection and Relationship Classification](https://arxiv.org/abs/1912.05135)

Data
------
![alt text](https://github.com/ennauata/buildings2vec/blob/master/refs/raw.jpg "Raw images")
We borrow images from SpaceNet [1] corpus, which are hosted as an Amazon Web Services (AWS) Public Dataset and annotated 2,001 buildings across Los Angeles, Las Vegas and Paris, [click here to download](https://www.dropbox.com/sh/q1jmqnm26q21h1a/AABtxO0Uni9eZs-Qs37HJTJLa?dl=0) our annotations and cropped RGB images.<br/>
<br/>
[[1]](https://spacenetchallenge.github.io/datasets/datasetHomePage.html.) SpaceNet on Amazon Web Services (AWS). “Datasets.” The SpaceNet Catalog.  Last modified April 30, 2018. 

Running pretrained models
------
PS.: This code is still not clean nor optimized for good performance. Some modules were borrowed from Mask-RCNN, for more detailed documentation refer to https://github.com/facebookresearch/maskrcnn-benchmark. 

## Detecting corners primitives (PC) and corner-edge relationships (CE)

- Change paths in refs.py and datasets/junction.py
- Download pretrained model and move to output/{EXP_ID}/
- Pretrained model https://www.dropbox.com/s/h72ux3w32o6t9au/pretrained_junctions.zip?dl=0
- Run python3 main.py --exp 3 --json --test --checkepoch 15 --gpu 0
- Predictions should appear in result/

## Detecting edges primitives (PE)
- Change paths in detect.py
- Download pretrained model https://www.dropbox.com/s/b2dcqhb0de2xkua/pretrained_edges.zip?dl=0
- Run python3 detect.py
- Predictions will appear in ./output

## Detecting regions primitive (PR)

- Build maskrcnn
- Download pretrained model https://www.dropbox.com/s/wclufzt3liq120y/pretrained_regions.zip?dl=0
- Set paths in '/home/nelson/Workspace/outdoor_project_to_submit/region_detector/maskrcnn_benchmark/config/paths_catalog.py' 
- python3 ./tools/test_net.py --config-file '/home/nelson/Workspace/building_reconstruction/working_model/maskrcnn-boundary/configs/buildings_mask_rcnn_R_50_FPN_1x.yaml'


## Detecting region-to-region relationships (RR)

- Build maskrcnn
- Download pretrained model https://www.dropbox.com/s/9ju3iwwexecz69j/pretrained_shared_edges.zip?dl=0
- Set paths in '/home/nelson/Workspace/outdoor_project_to_submit/region_detector/maskrcnn_benchmark/config/paths_catalog.py' 
- python3 ./tools/test_net.py --config-file '/home/nelson/Workspace/building_reconstruction/working_model/maskrcnn-boundary/configs/buildings_mask_rcnn_R_50_FPN_1x.yaml'


## Ensembling primitives and relationships using IP

- Set paths in run_ablation_experiments.py
- Run python3 run_ablation_experiments.py
