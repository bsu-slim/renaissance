from renaissance.datamodules.datamodule_base import BaseDataModule
from renaissance.datasets.glue_dataset import GlueDataset

class GlueDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.hf_dataset_key = "glue"
        
        if self.config["loss_names"]["snli"] > 0:
            self.task = "snli"
        elif self.config["loss_names"]["mrpc"] > 0:
            self.task = "mrpc"
        elif self.config["loss_names"]["rte"] > 0:
            self.task = "rte" 
        elif self.config["loss_names"]["wnli"] > 0:
            self.task = "wnli"
        elif self.config["loss_names"]["sst2"] > 0:
            self.task = "sst2"
        elif self.config["loss_names"]["qqp"] > 0:
            self.task = "qqp"
        elif self.config["loss_names"]["qnli"] > 0:
            self.task = "qnli"
        elif self.config["loss_names"]["mnli"] > 0:
            self.task = "mnli"
        elif self.config["loss_names"]["cola"] > 0:
            self.task = "cola"
        else:
            raise ValueError("Selected task is not supported by GlueDataModule.")
        
    @property
    def dataset_cls(self):
        return GlueDataset

    @property
    def dataset_cls_no_false(self):
        return GlueDataset
    @property
    def dataset_name(self):
        return "glue"