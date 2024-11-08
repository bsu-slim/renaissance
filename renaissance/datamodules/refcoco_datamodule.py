from ..datasets.refcoco_dataset import RefcocoDataset
from .datamodule_base import BaseDataModule



class RefcocoDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return RefcocoDataset

    @property
    def dataset_cls_no_false(self):
        return RefcocoDataset

    @property
    def dataset_name(self):
        return "refcoco"