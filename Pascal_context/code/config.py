
"""ADE config system
"""
import numpy as np
import os.path
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# nyud setting
__C.superpixel = None
__C.rand_idx = None #superpixel index
__C.rand_coord = None # samplig coordinate
__C.N_sample = 1600

__C.batch = 2
__C.data_idx = None #input image index
__C.data_name = None #input image name
__C.width = 224
__C.height = 224
__C.replace_idx=None # replace superpixel index
__C.N_pixel = None
