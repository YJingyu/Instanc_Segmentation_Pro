
import collections
import glob
import json
import os
import shutil

import cv2
import numpy
from tqdm import tqdm
from pycocotools import mask as mask_util


data_dir = '/lengyu.yb/datasets/segmentation/MMSports22'
annotation_dir = 'basketball-instants-dataset/annotations'
label_dir = 'total_aug/labels'
image_dir = 'total_aug/images'

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mask_to_polygon(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = numpy.ascontiguousarray(mask)
    # some versions of cv2 does not support in contiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [(x + 0.5).tolist() for x in res if len(x) >= 6]
    return res, has_holes

def json2txt():
    print('Parsing ...')
    names = ['human']
    json_files = glob.glob(f'{data_dir}/{annotation_dir}/train.json')
    json_files2 = glob.glob(f'{data_dir}/{annotation_dir}/val.json')
    json_files3 = glob.glob(f'{data_dir}/{annotation_dir}/test.json')
    json_files.extend(json_files2)
    json_files.extend(json_files3)

    segmentation = collections.defaultdict(list)
    for json_file in sorted(json_files):
        with open(json_file) as f:
            json_data = json.load(f)
        images = json_data['images']
        id2image_name = {}
        for image in images:
            file_name = image['file_name'].split('/')[-1]
            id2image_name[image['id']] = file_name
        for annotation in tqdm(json_data['annotations']):
            file_name = id2image_name[annotation['image_id']]
            mask = mask_util.decode(annotation['segmentation'])
            polygons, has_hole = mask_to_polygon(mask)
            if len(polygons) > 0:
                seg_item = collections.defaultdict(list)
                for polygon in polygons:
                    poly = []
                    p = polygon
                    for i in range(0, len(p), 2):
                        poly.append([int(p[i]), int(p[i + 1])])
                    x, y, w, h = list(map(int, cv2.boundingRect(numpy.array([poly], int))))
                    area = w * h
                    seg_item[area].append(polygon)
                max_area = max(seg_item.keys())
                polys = seg_item[max_area]
                poly = []
                p = polys[0]
                if len(p):
                    for i in range(0, len(p), 2):
                        poly.append([int(p[i]), int(p[i + 1])])
                points = [names[annotation["category_id"] - 1]]
                for p in poly:
                    points.append(p[0])
                    points.append(p[1])
                points = " ".join([str(p) for p in points])
                segmentation[file_name].append(points)

    for key, value in segmentation.items():
        with open(f'{data_dir}/{label_dir}/{key[:-4]}.txt', 'w') as f:
            for v in value:
                f.write(f'{v}\n')
    
    json_file =f'{data_dir}/{annotation_dir}/train.json'
    with open(json_file) as f:
        json_data = json.load(f)
    images = json_data['images']
    json_file =f'{data_dir}/{annotation_dir}/val.json'
    with open(json_file) as f:
        json_data = json.load(f)
    images2 = json_data['images']
    json_file =f'{data_dir}/{annotation_dir}/test.json'
    with open(json_file) as f:
        json_data = json.load(f)
    images3 = json_data['images']
    images.extend(images2)
    images.extend(images3)
    for filename in [image['file_name'] for image in images]:
        dst_name = filename.split('/')[-1]
        shutil.copyfile(f'{data_dir}/basketball-instants-dataset/{filename}', f'{data_dir}/{image_dir}/{dst_name}')
    
make_dir(os.path.join(data_dir, image_dir))
make_dir(os.path.join(data_dir, label_dir))
json2txt()