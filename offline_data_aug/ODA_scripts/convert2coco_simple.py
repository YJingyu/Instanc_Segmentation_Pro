import collections
import glob
import json
import os
import shutil

import cv2
import numpy
import tqdm
from pycocotools import mask as mask_util


data_dir = '/lengyu.yb/datasets/segmentation/MMSports22'

# annotation_dir = 'total_simple.json'
# aug_image_dir = 'total_images_simple'
# aug_label_dir = 'total_labels_simple'

annotation_dir = 'trainval_simple.json'
aug_image_dir = 'total_images_simple'
aug_label_dir = 'trainval_labels_simple'


def convert2coco():
    print('Converting into COCO ...')
    classes = ('human',)
    filenames = [filename for filename in os.listdir(f'{data_dir}/{aug_label_dir}')]
    img_id = 0
    box_id = 0
    images = []
    categories = []
    annotations = []
    for filename in tqdm.tqdm(filenames):
        img_id += 1
        h, w = cv2.imread(f"{data_dir}/{aug_image_dir}/{filename[:-4]}.png").shape[:2]
        images.append({'file_name': f"{filename[:-4]}.png", 'id': img_id, 'height': h, 'width': w})
        regions = []
        with open(f'{data_dir}/{aug_label_dir}/{filename}') as f:
            for line in f.readlines():
                regions.append(line.rstrip())
        for region in regions:
            box_id += 1
            region = region.split(' ')
            mask = region[1:]
            poly = []
            for i in range(0, len(mask), 2):
                poly.append([int(mask[i]), int(mask[i + 1])])
            x_min, y_min, w, h = cv2.boundingRect(numpy.array([poly], int))
            bbox = [x_min, y_min, w, h]

            category_id = classes.index(region[0])
            annotations.append({'id': box_id,
                                'bbox': bbox,
                                'iscrowd': 0,
                                'image_id': img_id,
                                'segmentation': [list(map(int, mask))],
                                'area': bbox[2] * bbox[3],
                                'category_id': category_id})
    for category_id, category in enumerate(classes):
        categories.append({'supercategory': category, 'id': category_id, 'name': category})
    print(len(images), 'images')
    print(len(annotations), 'instances')
    json_data = json.dumps({'images': images, 'categories': categories, 'annotations': annotations})
    with open(f'{data_dir}/{annotation_dir}', 'w') as f:
        f.write(json_data)


convert2coco()

## total
# 3240 images
# 24040 instances
## trainval
# 2600 images
# 19370 instances