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
annotation_dir = 'basketball-instants-dataset/annotations'
label_dir = 'total_aug/labels'
image_dir = 'total_aug/images'

crop_image_dir = 'total_aug/c_images'
crop_label_dir = 'total_aug/c_labels'

source_image_dir = 'total_aug/src_images'
source_label_dir = 'total_aug/src_labels'

source_image_dir = 'total_aug/src_images'
aug_image_dir = 'total_aug/aug_images'
aug_label_dir = 'total_aug/aug_labels'

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def add_object_fn(dst_name):
    dst_img = cv2.imread(dst_name)

    if '0_aug' not in dst_name:
        dst_points = []
        filename = os.path.basename(dst_name).split(':')[0]
        with open(f'{data_dir}/{label_dir}/{filename}.txt') as f:
            for line in f.readlines():
                line = line.rstrip().split(' ')
                point = list(map(int, line[1:]))
                point.insert(0, line[0])
                point = " ".join([str(p) for p in point])
                dst_points.append(point)
    else:
        with open(f'{data_dir}/{crop_label_dir}/0.txt') as f:
            labels = {}
            for line in f.readlines():
                line = line.rstrip().split(' ')
                labels[line[0]] = line[1:]

        dst_h, dst_w = dst_img.shape[:2]

        poly = []
        dst_poly = []
        src_name = f'{data_dir}/{crop_image_dir}/0/human_99.png'
        label = labels[os.path.basename(src_name)]
        src_img = cv2.imread(src_name)
        for i in range(0, len(label), 2):
            poly.append([int(label[i]), int(label[i + 1])])
        src_mask = numpy.zeros(src_img.shape, src_img.dtype)
        cv2.fillPoly(src_mask, [numpy.array(poly)], (255, 255, 255))
        x_c, y_c = dst_w // 2, dst_h // 2
        for p in poly:
            dst_poly.append([int(p[0] + x_c), int(p[1] + y_c)])
        dst_mask = numpy.zeros(dst_img.shape, dst_img.dtype)
        cv2.fillPoly(dst_mask, [numpy.array(dst_poly, int)], (255, 255, 255))
        h, w = src_img.shape[:2]
        dst_points = []

        dst_point = ['human']
        for p in dst_poly:
            dst_point.append(p[0])
            dst_point.append(p[1])
        dst_point = " ".join([str(p) for p in dst_point])
        dst_points.append(dst_point)
        dst_img[dst_mask > 0] = 0
        dst_img[y_c:y_c + h, x_c:x_c + w] += src_img * (src_mask > 0)
    with open(f'{data_dir}/{aug_label_dir}/{os.path.basename(dst_name)[:-4]}.txt', 'w') as f:
        for dst_point in dst_points:
            f.write(f'{dst_point}\n')
    cv2.imwrite(f'{data_dir}/{aug_image_dir}/{os.path.basename(dst_name)}', dst_img)

def add_object():
    print('Augmenting ...')
    import multiprocessing
    folders = [folder for folder in os.listdir(f'{data_dir}/{source_image_dir}')]
    for folder in folders:
        if not folder.endswith('aug'):
            continue
        dst_names = glob.glob(f'{data_dir}/{source_image_dir}/{folder}/*.png')
        with multiprocessing.Pool(os.cpu_count() - 4) as pool:
            pool.map(add_object_fn, dst_names)
        pool.close()

make_dir(os.path.join(data_dir, aug_image_dir))
make_dir(os.path.join(data_dir, aug_label_dir))
add_object()