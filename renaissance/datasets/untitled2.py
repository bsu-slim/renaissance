#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:15:33 2024

@author: claytonfields
"""

import json
import pandas as pd
import pyarrow as pa
import os

import sys
import os.path as osp
import json
import pickle
import time
import itertools
# import skimage.io as io


from PIL import Image

import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt

from torchvision import transforms
import torch
# import matplotlib.pyplot as plt
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Polygon, Rectangle
from pprint import pprint
import numpy as np
from refer import REFER

from tqdm import tqdm
from collections import defaultdict


def get_bounded_subimage(refer, img_id, ann_id, xs=224,ys=224, show=False):
    bbox = refer.Anns[ann_id]['bbox']
    bbox = [int(b) for b in bbox]
    img = refer.Imgs[img_id]
    I = skio.imread(os.path.join(refer.IMAGE_DIR, img['file_name']))
    sub = I[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    if show:
        plt.figure()
        ax = plt.gca()
        ax.imshow(sub)
        plt.show()
    if len(sub) == 0: return None
    pim = Image.fromarray(sub)
    pim2 = pim.resize((xs,ys), Image.ANTIALIAS)
    # img = np.array(pim2)
    img = transforms.functional.to_tensor(pim2)
    if len(img.shape) < 3: return None
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    return img
    

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()
    
data_root = '/home/claytonfields/nlp/code/data/coco'  # contains refclef, refcoco, refcoco+, refcocog and images
dataset = 'refcoco' 
splitBy = 'unc'

print ('loading dataset %s into memory...' % dataset)
print('testing')
# ROOT_DIR = osp.abspath(osp.dirname(__file__))
DATA_DIR = osp.join(data_root, dataset)
if dataset in ['refcoco', 'refcoco+', 'refcocog']:
    IMAGE_DIR = osp.join(data_root, 'images/mscoco/train2014')
elif dataset == 'refclef':
    IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
else:
    print ('No refer dataset is called [%s]' % dataset)
    sys.exit()

# load refs from data/dataset/refs(dataset).json
tic = time.time()
ref_file = osp.join(DATA_DIR, 'refs('+splitBy+').p')
data = {}
data['dataset'] = dataset
data['refs'] = pickle.load(StrToBytes(open(ref_file, 'r')))

# load annotations from data/dataset/instances.json
instances_file = osp.join(DATA_DIR, 'instances.json')
instances = json.load(open(instances_file, 'r'))
data['images'] = instances['images']
data['annotations'] = instances['annotations']
data['categories'] = instances['categories']

# # create index
# createIndex()
# print ('DONE (t=%.2fs)' % (time.time()-tic)