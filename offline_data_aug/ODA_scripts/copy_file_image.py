
import collections
import glob
import json
import os
import shutil

import cv2
import numpy
from tqdm import tqdm
import random, colorsys


data_dir = '/lengyu.yb/datasets/segmentation/MMSports22'


annotation_dir = 'basketball-instants-dataset/annotations'
label_dir = 'total_labels_simple'
image_dir = 'total_images_simple'


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_file():
    dst_names = set(name.split(':')[0] for name in os.listdir(f'{data_dir}/{source_image_dir}'))
    if not os.path.exists(f'{data_dir}/{destat_image_dir}'):
        make_dir(f'{data_dir}/{destat_image_dir}')
    for dst_name in tqdm(dst_names):
        for i in range(1, 11, 1):
            image_path = f'{data_dir}/{source_image_dir}/{dst_name}:1.png'
            save_path = f'{data_dir}/{destat_image_dir}/{dst_name}:{i}.png'
            shutil.copyfile(image_path, save_path)
copy_file()

# nohup python /knt/lengyu.yb/datasets/segmentation/instance_segmentation/vipriors-segmentation-data-2022/scripts/copy_file.py >> out.log 2>&1 &