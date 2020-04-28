import numpy as np
from pathlib import Path
import os

home = Path.home()

eps = 1e-6
momentum = 0.9
weightDecay = 0.0
alpha = 0.99
epsilon = 1e-8

nThreads = 4
pixel_mean = np.array([115.9839754, 126.63120922, 137.73309306], dtype=np.float32)

ext = '.pickle'
#ext = '.pk'
data_dir = Path('/home/nelson/Workspace/cities_dataset')
root_dir = Path(os.getcwd()) / '..'
data_root = data_dir
junc_data_root = data_root / 'PC_and_CE_inference'
output_root = root_dir / "output/"
result_dir = root_dir / 'result' / 'PC_and_CE_inference'
hypeDir = root_dir / 'PC_and_CE_inference/hypes'
logdir = root_dir / 'logs'
