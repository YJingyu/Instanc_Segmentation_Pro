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

source_image_dir = 'total_aug/src_images'

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sort_images():
    print('Sorting ...')
    label_names = [filename[:-4] for filename in os.listdir(f'{data_dir}/{label_dir}')]
    image_names = [filename[:-4] for filename in os.listdir(f'{data_dir}/{image_dir}')]
    for image_name in tqdm(image_names):
        if image_name in label_names:
            with open(f'{data_dir}/{label_dir}/{image_name}.txt') as f:
                num_objects = len(f.readlines())
            if not os.path.exists(f'{data_dir}/{source_image_dir}/{str(num_objects)}'):
                os.makedirs(f'{data_dir}/{source_image_dir}/{str(num_objects)}')
            shutil.copyfile(f'{data_dir}/{image_dir}/{image_name}.png',
                            f'{data_dir}/{source_image_dir}/{str(num_objects)}/{image_name}.png')
        else:
            if not os.path.exists(f'{data_dir}/{source_image_dir}/0'):
                os.makedirs(f'{data_dir}/{source_image_dir}/0')
            shutil.copyfile(f'{data_dir}/{image_dir}/{image_name}.png',
                            f'{data_dir}/{source_image_dir}/0/{image_name}.png')

make_dir(os.path.join(data_dir, source_image_dir))
sort_images()