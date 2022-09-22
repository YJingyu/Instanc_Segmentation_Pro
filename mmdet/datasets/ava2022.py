from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module
class AVA2022Dataset(CocoDataset):

    CLASSES = ('nondisabledped', 'wheelchairped', 'visuallyimpairedped', 'rider', 
                'fourwheelveh', 'twowheelveh', 'wheelchair', 'cane')