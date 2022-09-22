import os, json
import pycocotools.mask as mask_util
from skimage import morphology,measure
from tqdm import tqdm


htc_file1 = '/lengyu.yb/models/ava/submission_leaderboard/submission12_61.93.json'
htc_file2 = '/lengyu.yb/models/ava/submission_leaderboard/submission13_61.46.json'
msk_file = '/lengyu.yb/models/ava/mask2former/submission1_47.92.json'
## 0.01/62.59975252983095
## w/62.60049133487946
dst_file = '/lengyu.yb/models/ava/final/submission2.json'

htc_json1 = json.load(open(htc_file1))
htc_json2 = json.load(open(htc_file2))
msk_json = json.load(open(msk_file))

refine_json = []
for i in tqdm(range(len(htc_json1))):
    category_id = htc_json1[i]['category_id']
    if category_id != 8 and category_id != 5 and category_id != 6:
        refine_json.append(htc_json1[i])

for i in tqdm(range(len(htc_json2))):
    category_id = htc_json2[i]['category_id']
    if category_id == 5 or category_id == 6:
        refine_json.append(htc_json2[i])

for i in tqdm(range(len(msk_json))):
    category_id = msk_json[i]['category_id']
    if category_id==8:
        # if msk_json[i]['score']>0.01:
        refine_json.append(msk_json[i])

print('total annotations is ', len(refine_json))
with open(dst_file, 'w') as tfile:
    json.dump(refine_json, tfile)