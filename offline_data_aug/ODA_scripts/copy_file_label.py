
import os
import shutil

from tqdm import tqdm


data_dir = '/lengyu.yb/datasets/segmentation/MMSports22'


# # total-tain val test
# source_image_dir = 'total_aug/aug_labels'
# destat_image_dir = 'total_labels_simple'

# # tain val
source_image_dir = 'trainval_aug/aug_labels'
destat_image_dir = 'trainval_labels_simple'


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_file():
    dst_names = set(name.split(':')[0] for name in os.listdir(f'{data_dir}/{source_image_dir}'))
    if not os.path.exists(f'{data_dir}/{destat_image_dir}'):
        make_dir(f'{data_dir}/{destat_image_dir}')
    for dst_name in tqdm(dst_names):
        for i in range(1, 11, 1):
            image_path = f'{data_dir}/{source_image_dir}/{dst_name}:1.txt'
            save_path = f'{data_dir}/{destat_image_dir}/{dst_name}:{i}.txt'
            shutil.copyfile(image_path, save_path)

copy_file()
