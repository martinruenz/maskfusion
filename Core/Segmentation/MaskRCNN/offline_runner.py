#!/usr/bin/env python3
#
# This file is part of https://github.com/martinruenz/maskfusion
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

# To use this script, add the MaskRCNN directoy to your PYTHON_PATH
import sys
import os

mask_rcnn_path = os.path.abspath("../../../deps/Mask_RCNN")
sys.path.insert(0, mask_rcnn_path)

import random
import math
import numpy as np
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import argparse
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from PIL import Image
from helpers import *
import time
import pytoml as toml
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument("-i", required=True, help="Input directory (all files are being processed)")
parser.add_argument("-c", required=False, help="Optional config file, otherwise MsCoco is assumed")
parser.add_argument("-o", required=True, help="Output directory")
parser.add_argument("--filter", nargs='+', required=False,
                    help="Specify which labels you would like to use (empty means all), example: --filter teddy_bear pizza baseball_bat")
args = parser.parse_args()

# FURTHER PARAMETERS
EXTENSIONS = ['jpg', 'png']
FILTER_IMAGE_NAME = ""  # only use images, whose name contains this string (eg "Color")
score_threshold = 0.85
SPECIAL_ASSIGNMENTS = {} #{'person': 255}
SINGLE_INSTANCES = False
OUTPUT_FRAMES = True
STORE_CLASS_IDS = True
START_INDEX = 0

IMAGE_DIR = args.i
OUTPUT_DIR = args.o
DATA_DIR = os.path.join(mask_rcnn_path, "data")
MODEL_DIR = os.path.join(DATA_DIR, "logs")
model_path = os.path.join(DATA_DIR, "mask_rcnn_coco.h5")

filter_classes = []
if args.filter:
    filter_classes = args.filter
    filter_classes = [f.replace("_", " ") for f in filter_classes]
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

if args.c:
    with open(args.c, 'rb') as toml_file:
        toml_config = toml.load(toml_file)
        class_names = toml_config["MaskRCNN"]["class_names"]
        model_path = toml_config["MaskRCNN"]["model_path"]
        filter_classes = toml_config["MaskRCNN"]["filter_classes"]
        score_threshold = toml_config["MaskRCNN"]["score_threshold"]

filter_classes = [class_names.index(x) for x in filter_classes]
SPECIAL_ASSIGNMENTS = {class_names.index(x): SPECIAL_ASSIGNMENTS[x] for x in SPECIAL_ASSIGNMENTS}

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(class_names)

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(model_path, by_name=True)

file_names = [fn for fn in os.listdir(IMAGE_DIR) if any(fn.endswith(ext) for ext in EXTENSIONS)]
file_names.sort()
if FILTER_IMAGE_NAME and FILTER_IMAGE_NAME != "":
    file_names = [fn for fn in file_names if FILTER_IMAGE_NAME in fn]

# ALL TOGETHER:
# print("Loading images...")
# loaded_images = [scipy.misc.imread(os.path.join(IMAGE_DIR, f)) for f in file_names]
# print("Starting evaluation...")
# start_time = time.time()
# results = model.detect(loaded_images, verbose=0)
# duration = time.time() - start_time
# print("Evaluation took {} seconds.".format(duration))
# for idx, result in enumerate(results):
#     out_path = os.path.join("/tmp/test", "{}.png".format(idx))
#     output_mask_ids(result, out_path)


# SEPARATELY
fig = plt.figure()
ax = fig.add_subplot(111)
# plt.show(block=False)
plt.ion()
#_, ax = plt.subplots(1, figsize=(16, 16))
for idx, file_name in enumerate(file_names):

    if idx < START_INDEX:
        continue

    base_name = str(idx).zfill(4)

    if os.path.isfile(os.path.join(OUTPUT_DIR, base_name + ".png")):
        continue

    print("Starting to work on frame", base_name)

    image = scipy.misc.imread(os.path.join(IMAGE_DIR, file_name))
    h, w = image.shape[:2]

    results = model.detect([image], verbose=0)
    r = results[0]

    if len(r['class_ids']) == 0:
        r['masks'] = np.empty(shape=[h, w, 0])
        r['scores'] = []
        r['class_ids'] = []
        r['rois'] = np.empty(shape=[0, 4])

    if SINGLE_INSTANCES:
        merge_instances(r)

    #out_path = os.path.join(OUTPUT_DIR, "{}.png".format(idx))
    id_image, exported_class_ids, exported_rois = generate_id_image(r, score_threshold, filter_classes, SPECIAL_ASSIGNMENTS)
    save_id_image(id_image, OUTPUT_DIR, base_name, exported_class_ids, STORE_CLASS_IDS, exported_rois)


    # Visualise
    ax.clear()
    filter_result(r, filter_classes)
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                            class_names, r['scores'], score_threshold, ax=ax) # requires patched version
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], ax=ax)
    fig.canvas.draw()
    if OUTPUT_FRAMES:
        plt.savefig(os.path.join(OUTPUT_DIR, base_name+".jpg"))
