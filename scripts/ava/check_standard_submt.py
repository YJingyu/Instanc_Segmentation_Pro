import os, json
import pycocotools.mask as mask_util
from skimage import morphology,measure
from tqdm import tqdm



# result_file = '/lengyu.yb/models/ava/final/submission.json'
result_file = '/lengyu.yb/models/ava/submission_leaderboard/submission12_61.93.json'


result_json = json.load(open(result_file))
image_ids = []
for i in tqdm(range(len(result_json))):
    image_id = result_json[i]['image_id']
    image_ids.append(image_id)
    
image_ids_set = set(image_ids)
print('total id ', len(image_ids_set))
print('id sample: ', list(image_ids_set)[:5])