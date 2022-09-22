'''
对copy paste的人/球做筛选，面积太大的人不进行复制粘帖了
'''
import os
import cv2
from tqdm import tqdm
import shutil


data_dir = '/lengyu.yb/datasets/segmentation/VIPriors2022/datasets'

crop_image_dir = 'images_crop'
crop_label_dir = 'labels_crop'

dst_crop_image_dir = 'images_crop_sel2'
dst_crop_label_dir = 'labels_crop_sel2'


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


select_human = []
# 过滤掉面积太大的人
for item in tqdm(os.listdir(f'{data_dir}/{crop_image_dir}/0')):
    img_path = os.path.join(f'{data_dir}/{crop_image_dir}/0/{item}')
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    if w > 200 or h > 200:  # 1249
        continue
    else:
        select_human.append(item)
        dst_path = os.path.join(f'{data_dir}/{dst_crop_image_dir}/0/{item}')
        shutil.copy(img_path, dst_path)
        # print(img.shape, item)
print(len(select_human))
# 1897->1249
