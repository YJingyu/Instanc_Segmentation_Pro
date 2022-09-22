import os, json
import pycocotools.mask as mask_util
from skimage import morphology,measure
from tqdm import tqdm


# htc_file1 = '/lengyu.yb/models/ava/submission_leaderboard/submission12_61.93.json'
# htc_file2 = '/lengyu.yb/models/ava/submission_leaderboard/submission13_61.46.json'
# htc_file3 = '/lengyu.yb/models/ava/submission_leaderboard/submission15_60.56.json'
# msk_file = '/lengyu.yb/models/ava/mask2former/submission2_50.78.json'
# 62.97583778

htc_file1 = '/lengyu.yb/models/ava/submission_leaderboard/submission12_61.93.json'
htc_file2 = '/lengyu.yb/models/ava/submission_leaderboard/submission18_61.78.json'
htc_file3 = '/lengyu.yb/models/ava/submission_leaderboard/submission15_60.56.json'
msk_file = '/lengyu.yb/models/ava/mask2former/submission2_50.78.json'


dst_file = '/lengyu.yb/models/ava/final/submission.json'

htc_json1 = json.load(open(htc_file1))
htc_json2 = json.load(open(htc_file2))
htc_json3 = json.load(open(htc_file3))
msk_json = json.load(open(msk_file))

refine_json = []
for i in tqdm(range(len(htc_json1))):
    category_id = htc_json1[i]['category_id']
    if category_id != 8 and category_id != 5 and category_id != 6:
        refine_json.append(htc_json1[i])

for i in tqdm(range(len(htc_json2))):
    category_id = htc_json2[i]['category_id']
    if category_id == 6:
        refine_json.append(htc_json2[i])

for i in tqdm(range(len(htc_json3))):
    category_id = htc_json3[i]['category_id']
    if category_id == 5:
        refine_json.append(htc_json3[i])

for i in tqdm(range(len(msk_json))):
    category_id = msk_json[i]['category_id']
    if category_id==8:
        refine_json.append(msk_json[i])

print('total annotations is ', len(refine_json))
with open(dst_file, 'w') as tfile:
    json.dump(refine_json, tfile)