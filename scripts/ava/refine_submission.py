import os, json
import pycocotools.mask as mask_util
from skimage import morphology,measure
from tqdm import tqdm


# htc_file = '/lengyu.yb/models/ava/submission_leaderboard/submission12_61.93.json'
# msk_file = 'lengyu.yb/models/segmentation/InstanceSegmentation/Mask2Former/submission/submission1-40.01.json'
## 0.5/0.05 61.94904581592212
## w/0.01 62.067053758951396

htc_file = '/lengyu.yb/models/ava/submission_leaderboard/submission12_61.93.json'
msk_file = '/lengyu.yb/models/ava/mask2former/submission1_47.92.json'
## 融合策略2: 59.01324231668621
## 融合策略1: 
dst_file = '/lengyu.yb/models/ava/final/submission.json'

htc_json = json.load(open(htc_file))
msk_json = json.load(open(msk_file))

## 融合策略2
# refine_json = []
# for i in tqdm(range(len(htc_json))):
#     category_id = htc_json[i]['category_id']
#     if category_id==8:
#         if htc_json[i]['score']>0.5:
#             refine_json.append(htc_json[i])
#     else:
#         refine_json.append(htc_json[i])

# for i in tqdm(range(len(msk_json))):
#     category_id = msk_json[i]['category_id']
#     if category_id==8:
#         if msk_json[i]['score']>0.01:
#             refine_json.append(msk_json[i])
#     else:
#         if msk_json[i]['score']>0.8:
#             refine_json.append(msk_json[i])

## 融合策略1
refine_json = []
# for i in tqdm(range(len(htc_json))):
#     category_id = htc_json[i]['category_id']
#     if category_id==8:
#         if htc_json[i]['score']>0.5:
#             refine_json.append(htc_json[i])
#     else:
#         refine_json.append(htc_json[i])

for i in tqdm(range(len(htc_json))):
    category_id = htc_json[i]['category_id']
    if category_id != 8:
        refine_json.append(htc_json[i])

for i in tqdm(range(len(msk_json))):
    category_id = msk_json[i]['category_id']
    if category_id==8:
        if msk_json[i]['score']>0.01:
            refine_json.append(msk_json[i])

print('total annotations is ', len(refine_json))
with open(dst_file, 'w') as tfile:
    json.dump(refine_json, tfile)