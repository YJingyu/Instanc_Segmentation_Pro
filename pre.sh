# pytorch:1.7.0
git clone https://github.com/NVIDIA/apex
cd apex
rm -rf build
python setup.py install --cuda_ext --cpp_ext

pip install mmcv==1.3.8
pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip uninstall mmdet -y 
rm -rf build
# python setup.py install
python setup.py develop
cd cocoapi/pycocotools
pip install .
