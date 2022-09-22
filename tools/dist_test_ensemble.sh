#!/usr/bin/env bash
CONFIG='configs/ava2022/htc_cbv2_swin_large_giou_4conv1f_ava2022.py'
CHECKPOINT='/lengyu.yb/models/ava/cbv2/htc_wsemantic_swin1/epoch_17.pth'
GPUS=4
PORT=${PORT:-29600}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_ensemble.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} \
        --format-only --options 'jsonfile_prefix=./submission/submission'