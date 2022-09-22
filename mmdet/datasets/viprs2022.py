from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module
class VIPrs2022Dataset(CocoDataset):

    CLASSES = ('human', 'ball')