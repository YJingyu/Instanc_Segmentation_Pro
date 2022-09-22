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

crop_image_dir = 'total_aug/c_images'
crop_label_dir = 'total_aug/c_labels'


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

def crop_objects():
    print('Cropping ...')
    names = ['human']
    counter = collections.defaultdict(int)
    name_set = collections.defaultdict(list)

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
            img = cv2.imread(f'{data_dir}/{image_dir}/' + file_name)
            if not os.path.exists(f'{data_dir}/{crop_image_dir}/' + str(annotation['category_id'])):
                os.makedirs(f'{data_dir}/{crop_image_dir}/' + str(annotation['category_id']))
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

                rect = cv2.boundingRect(numpy.array([poly], int))  # returns (x,y,w,h) of the rect
                crop = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
                label = annotation["category_id"]
                counter[label] += 1
                img_name = f'{names[label]}_{counter[label]}.png'
                points = [img_name]
                for p in poly:
                    points.append(p[0] - rect[0])
                    points.append(p[1] - rect[1])
                points = " ".join([str(p) for p in points])
                name_set[label].append(points)
                points = [names[label]]
                for p in poly:
                    points.append(p[0])
                    points.append(p[1])
                points = " ".join([str(p) for p in points])
                segmentation[file_name].append(points)
                cv2.imwrite(f'{data_dir}/{crop_image_dir}/{str(label)}/{img_name}', crop)

    if not os.path.exists(f'{data_dir}/{crop_label_dir}'):
        os.makedirs(f'{data_dir}/{crop_label_dir}')
    for key, value in name_set.items():
        with open(f'{data_dir}/{crop_label_dir}/{key}.txt', 'w') as f:
            for v in value:
                f.write(f'{v}\n')

make_dir(os.path.join(data_dir, crop_image_dir))
make_dir(os.path.join(data_dir, crop_label_dir))
crop_objects()