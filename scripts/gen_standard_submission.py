import os, json
from tqdm import tqdm


# test_file = '/lengyu.yb/datasets/segmentation/MMSports22/basketball-instants-dataset/annotations/test.json'
## challenge
test_file = '/lengyu.yb/datasets/segmentation/MMSports22/basketball-instants-dataset/annotations/challenge.json'
test_json = json.load(open(test_file))

## get image ids
image_ids = []
images = test_json['images']
for i in tqdm(range(len(images))):
    image_ids.append(images[i]['id'])
# print(image_ids)

image_ids_index = {}
for idx, item in enumerate(image_ids):
    image_ids_index[item] = idx
# print(image_ids_index)


result_file = './submission/submission.segm.json'
dst_file = './submission/submission.json'

result_json = json.load(open(result_file))
refine_json = []
for i in range(len(image_ids)):
    refine_json.append([[],[]])
# print(len(refine_json))
for i in tqdm(range(len(result_json))):
    image_id = result_json[i]['image_id']
    refine_image_index = image_ids_index[image_id]
    bbox = result_json[i]['bbox']
    bbox.append(result_json[i]['score'])
    mask = result_json[i]['segmentation']
    refine_json[refine_image_index][0].append(bbox)
    refine_json[refine_image_index][1].append(mask)

with open(dst_file, 'w') as tfile:
    json.dump(refine_json, tfile)