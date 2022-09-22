The code for ACM MMSports'22 Instance Segmentation challenge. The paper and technical report will be released soon.

The code is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [CBNetV2](https://github.com/VDIGPKU/CBNetV2).

# ACM MMSports'22 Instance Segmentation second challenge

## setup environments
sh pre.sh

## train model with multi-gpus
nohup bash tools/dist_train.sh configs/mmsports2022/exp02.py 4 >> /lengyu.yb/logs/mmsports2022/exp07.log 2>&1 &

## swa model with multi-gpus
nohup bash tools/dist_train.sh configs/mmsports2022/exp02_swa.py 4 >> /lengyu.yb/logs/mmsports2022/exp07_swa.log 2>&1 &

Our training logs and config are shown in **weights** folder

## inference
bash tools/dist_test.sh configs/mmsports2022/exp02.py /lengyu.yb/logs/mmsports2022/exp07_swa/swa_model_148_mms.pth 4

## generate standard submission format
python scripts/gen_standard_submission.py

## evaluate on test set
python evaluate.py /lengyu.yb/datasets/segmentation/MMSports22/basketball-instants-dataset/annotations/test.json ./submission/submission.segm json ./
