
import collections
import glob
import json
import os
import shutil

import cv2
import numpy
from tqdm import tqdm
from pycocotools import mask as mask_util
from PIL import Image, ImageFilter
import random, colorsys

import augly.image as imaugs


data_dir = '/lengyu.yb/datasets/segmentation/MMSports22'
annotation_dir = 'basketball-instants-dataset/annotations'
label_dir = 'total_aug/labels'
image_dir = 'total_aug/images'

crop_image_dir = 'total_aug/c_images'
crop_label_dir = 'total_aug/c_labels'

source_image_dir = 'total_aug/src_images'
source_label_dir = 'total_aug/src_labels'

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_augmentations():
    AUGMENTATIONS = []
    # Color
    jitter_factor0 = random.randint(7,12)/10
    jitter_factor1 = random.randint(7,12)/10
    jitter_factor2 = random.randint(7,12)/10
    jitter_factor3 = random.randint(7,12)/10

    color_imaugs = [imaugs.OneOf([imaugs.RandomBrightness(min_factor=0.5, max_factor=1.5), 
                    imaugs.ColorJitter(jitter_factor1, jitter_factor2, jitter_factor3),
                    imaugs.Saturation(jitter_factor1)]), imaugs.Sharpen(jitter_factor0)]
    AUGMENTATIONS.extend(color_imaugs)

    # Quality
    opacity_factor = (random.random()+0.4)/2
    shuffle_factor = random.randint(0,2)/10
    # opacity_factor = opacity_factor if opacity_factor < 1 else 1
    # imaugs.Opacity(level=opacity_factor),
    quality_imaugs = [imaugs.OneOf([imaugs.RandomBlur(min_radius=0,max_radius=5), 
                        imaugs.RandomNoise(seed=random.randint(1,100)), 
                        imaugs.ShufflePixels(shuffle_factor)]),imaugs.RandomPixelization(min_ratio=0.2,max_ratio=0.6)]
    AUGMENTATIONS.extend(quality_imaugs)
    # Filter
    image_filter = random.sample([ImageFilter.DETAIL,
                ImageFilter.EDGE_ENHANCE,ImageFilter.SMOOTH,
                ImageFilter.MedianFilter,ImageFilter.ModeFilter],1)[0]
    filter_imaugs = imaugs.ApplyPILFilter(image_filter)
    AUGMENTATIONS.extend([filter_imaugs])

    return AUGMENTATIONS

def change_color_hue(image):
    # target_hue = random.randint(0,355)
    # target_hue = random.sample([0,120,240,355],1)[0]
    target_hue = random.random()
    image.load()	
    r, g, b = image.split()	
    result_r, result_g, result_b = [], [], []	
    for pixel_r, pixel_g, pixel_b in zip(r.getdata(), g.getdata(), b.getdata()):	
        h, s, v = colorsys.rgb_to_hsv(pixel_r / 255., pixel_b / 255., pixel_g / 255.)	
        rgb = colorsys.hsv_to_rgb(target_hue, s, v)	
        pixel_r, pixel_g, pixel_b = [int(x * 255.) for x in rgb]	
        result_r.append(pixel_r)	
        result_g.append(pixel_g)	
        result_b.append(pixel_b)		
    r.putdata(result_r)	
    g.putdata(result_g)	
    b.putdata(result_b)	
    image = Image.merge('RGB', (r, g, b))
    return image

def copy_file():
    print('Copying ...')
    folders = [folder for folder in os.listdir(f'{data_dir}/{source_image_dir}')]
    for folder in tqdm(folders):
        dst_names = [name[:-4] for name in os.listdir(f'{data_dir}/{source_image_dir}/{folder}')]
        if not os.path.exists(f'{data_dir}/{source_image_dir}/{folder}_aug'):
            os.makedirs(f'{data_dir}/{source_image_dir}/{folder}_aug')
            os.makedirs(f'{data_dir}/{source_label_dir}/{folder}_aug')
        for dst_name in dst_names:
            shutil.copyfile(f'{data_dir}/{source_image_dir}/{folder}/{dst_name}.png',
                            f'{data_dir}/{source_image_dir}/{folder}_aug/{dst_name}:1.png')
            if str(folder) != '0':
                shutil.copyfile(f'{data_dir}/{label_dir}/{dst_name}.txt',
                                f'{data_dir}/{source_label_dir}/{folder}_aug/{dst_name}:1.txt')
            for i in range(2, 21, 1):
                image_path = f'{data_dir}/{source_image_dir}/{folder}/{dst_name}.png'
                image = Image.open(image_path)
                augmentations = get_augmentations()
                for augmentation in augmentations:
                    image = augmentation(image)
                image = change_color_hue(image)
                save_file = f'{data_dir}/{source_image_dir}/{folder}_aug/{dst_name}:{i}.png'
                image.save(save_file)
                if str(folder) != '0':
                    shutil.copyfile(f'{data_dir}/{label_dir}/{dst_name}.txt',
                                    f'{data_dir}/{source_label_dir}/{folder}_aug/{dst_name}:{i}.txt')

make_dir(os.path.join(data_dir, source_label_dir))
copy_file()