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

Junctions

- Change paths in refs.py and datasets/junction.py
- Download pretrained model and move to output/{EXP_ID}/
- Run python3 main.py --exp 3 --json --test --checkepoch 15 --gpu 0
- Predictions should appear in result/

Edges
- Change paths in detect.py
- Run python3 detect.py
- Predictions will appear in ./output

Regions

- Build maskrcnn
- Download pretrained model
- Set paths in '/home/nelson/Workspace/outdoor_project_to_submit/region_detector/maskrcnn_benchmark/config/paths_catalog.py' 
- python3 ./tools/test_net.py --config-file '/home/nelson/Workspace/building_reconstruction/working_model/maskrcnn-boundary/configs/buildings_mask_rcnn_R_50_FPN_1x.yaml'


Shared edges

- Build maskrcnn
- Download pretrained model
- Set paths in '/home/nelson/Workspace/outdoor_project_to_submit/region_detector/maskrcnn_benchmark/config/paths_catalog.py' 
- python3 ./tools/test_net.py --config-file '/home/nelson/Workspace/building_reconstruction/working_model/maskrcnn-boundary/configs/buildings_mask_rcnn_R_50_FPN_1x.yaml'


IP optimizer

- Set paths in run_ablation_experiments.py
- Run python3 run_ablation_experiments.py
