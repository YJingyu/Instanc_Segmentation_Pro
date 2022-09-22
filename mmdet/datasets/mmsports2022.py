from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module
class MMSports2022Dataset(CocoDataset):

    CLASSES = ('human',)