from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cPickle as pickle
import cv2
import os
import sys

from detectron.utils.collections import AttrDict
from collections import OrderedDict
from detectron.datasets.json_dataset import JsonDataset
from detectron.core.config import cfg
from detectron.utils.logging import send_email
import detectron.datasets.cityscapes_json_dataset_evaluator \
    as cs_json_dataset_evaluator
import detectron.datasets.json_dataset_evaluator as json_dataset_evaluator
import detectron.datasets.voc_dataset_evaluator as voc_dataset_evaluator
import detectron.utils.vis as vis_utils
from detectron.datasets.task_evaluation import evaluate_all
import detectron.datasets.dataset_catalog as dataset_catalog
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from detectron.core.config import cfg
from detectron.utils.timer import Timer

with open('detection_range_0_5000.pkl', 'r') as f:
    dets = pickle.load(f)

assert all(k in dets for k in ['all_boxes', 'all_segms', 'all_keyps']), \
    'Expected detections pkl file in the format used by test_engine.py'

all_boxes = dets['all_boxes']
print(len(dets['all_boxes']))
all_segms = dets['all_segms']
all_keyps = dets['all_keyps']
dataset=AttrDict()
dataset.name='coco_2014_minival'
classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
dataset.classes = {i: name for i, name in enumerate(classes)}
dataset.image_directory= dataset_catalog.get_im_dir(dataset.name)
dataset.image_prefix = dataset_catalog.get_im_prefix(dataset.name)
dataset.COCO = COCO(dataset_catalog.get_ann_fn(dataset.name))
image_ids = (dataset.COCO.getImgIds())
image_ids.sort()
#print(image_ids[0:300])
dataset.debug_timer = Timer()
category_ids = dataset.COCO.getCatIds()
categories = [c['name'] for c in dataset.COCO.loadCats(category_ids)]
dataset.category_to_id_map = dict(zip(categories, category_ids))
dataset.classes = ['__background__'] + categories
dataset.num_classes = len(dataset.classes)
dataset.json_category_id_to_contiguous_id = {
    v: i + 1
    for i, v in enumerate(dataset.COCO.getCatIds())
}
dataset.contiguous_category_id_to_json_id = {
    v: k
    for k, v in dataset.json_category_id_to_contiguous_id.items()
}
print(evaluate_all(
    dataset, all_boxes, all_segms, all_keyps,'.', use_matlab=False
))
