import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.models.bert.modeling_bert import BertConfig#, BertModel, BertEmbeddings
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTConfig
from transformers.models.electra.modeling_electra import  ElectraConfig#,ElectraEmbeddings
from .embeddings import ElectraEmbeddings
# from transformers.model.vit import 
# from .bert_model import BertCrossLayer
from . import heads, objectives, renaissance_utils
from transformers import AutoConfig, AutoModel#, AutoModelForSequenceClassification
from .fusion_encoder import LxmertCrossModalEncoder
from .one_tower_encoder import OneTowerEncoder
from .two_tower_encoder import TwoTowerEncoder

class RenaissanceTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        # ===================== Base Architecture ===================== #
        self.model_type = config['model_type']
        # Adjust dimensions for fine-tuning
        self.fine_tune = (self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"])
        self.test_only = (self.hparams.config["load_path"] != "" 
            and self.hparams.config["test_only"])
        
        
        if self.fine_tune or self.test_only:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.original_max_text_len = ckpt['hyper_parameters']['config']['max_text_len']
            self.new_max_text_len = config['max_text_len']
            self.original_image_size = ckpt['hyper_parameters']['config']['original_image_size']
            self.new_image_size = config['image_size']
        
        if self.model_type == 'one-tower':
            self.pooler_type = config['pooler_type']
            
            if self.fine_tune or self.test_only:
                image_size = self.original_image_size
                max_text_len = self.original_max_text_len
            else:
                image_size = config['image_size']
                max_text_len = config['max_text_len']
            self.encoder = OneTowerEncoder(
                config,
                image_size,
                max_text_len,
                self.fine_tune,
                self.test_only
            )
            self.hidden_size = self.encoder.get_hidden_size()
            self.embedding_size = self.encoder.get_embedding_size()
        
        elif self.model_type == 'two-tower':
            self.encoder = TwoTowerEncoder(
                config,
                self.fine_tune,
                self.test_only
            )
            self.hidden_size = self.encoder.get_hidden_size()
            
        else:
            raise TypeError('Model Type not supported.')
        
        # ===================== Pretraining ===================== #
        
        if self.model_type =='one-tower':
            if self.pooler_type == 'single':
                hs = self.hs
            elif self.pooler_type == 'double':
                hs = 2*self.hidden_size
        else:
            hs = 2*self.hparams.config["cross_layer_hidden_size"]
        
        # Masked Language Modeling
        if self.hparams.config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(config, hidden_size=self.hidden_size)
            self.mlm_score.apply(objectives.init_weights)
        
        # Image Text Matching
        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(hs)
            self.itm_score.apply(objectives.init_weights)

        
        # ===================== Downstream  ===================== #
        
        # Initialize Visual Question Answering V2 Classifier
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = heads.MultiModalClassificationHead(
                hidden_size=hs, 
                num_labels=vs
            )
            self.vqa_classifier.apply(objectives.init_weights)
            
        # Load Previously Trained Modules
        if self.fine_tune:
            self.load_state_dict(state_dict, strict=False)
            if (self.model_type == 'one-tower') and (self.original_max_text_len != self.new_max_text_len):
                self.encoder.text_embeddings._adjust_position_embeddings(self.new_max_text_len)
            

        # Initialize NLVR2 Classifier
        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = heads.NLVR2ClassificationHead(
                hidden_size=hs, 
                num_labels=2
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            self.encoder.adjust_type_embeds_for_nlvr2()

        # Initialize SNLI-VE Classifier
        if self.hparams.config["loss_names"]["snli"] > 0:
            self.snli_classifier = heads.MultiModalClassificationHead(
                hidden_size=hs, 
                num_labels=3
            )
            self.snli_classifier.apply(objectives.init_weights)
            
        # Initialize Image-Text Recall Classifier
        # Possible error for two tower model below
        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(self.cross_layer_hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False
        
        # Initialize Reference Resolution Classifier
        if self.hparams.config["loss_names"]['ref'] > 0:
            self.ref_classifier = heads.MultiModalClassificationHead(
                hidden_size=hs, 
                num_labels=1
            )
            self.ref_classifier.apply(objectives.init_weights)
        
        # Text-Only Classification
        if self.model_type == 'one-tower':
            self.text_hs = self.hidden_size
        else:
            self.text_hs = config['text_encoder_hidden_size']
        
        
        self.text_only = False
     
        # MRPC Text Classifier
        if self.hparams.config["loss_names"]['mrpc'] > 0:
            # self.text_only = True
            # hidden_size = self.text_hs
            # num_labels = 2
            self.mrpc_classifier = heads.UniModalClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.mrpc_classifier.apply(objectives.init_weights)
            
        
        # rte Text Classifier
        if self.hparams.config["loss_names"]['rte'] > 0:
            # self.text_only = True
            # hidden_size = sel
            # num_labels = 2
            self.rte_classifier = heads.UniModalClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.rte_classifier.apply(objectives.init_weights)
        
        # wnli Text Classifier
        if self.hparams.config["loss_names"]['wnli'] > 0:
            # self.text_only = True
            self.wnli_classifier = heads.UniModalClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.wnli_classifier.apply(objectives.init_weights)
            
        # sst2 Text Classifier
        if self.hparams.config["loss_names"]['sst2'] > 0:
            # self.text_only = True
            self.sst2_classifier = heads.UniModalClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.sst2_classifier.apply(objectives.init_weights)
            
        # qqp Text Classifier
        if self.hparams.config["loss_names"]['qqp'] > 0:
            # self.text_only = True
            self.qqp_classifier = heads.UniModalClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.qqp_classifier.apply(objectives.init_weights)
            
        # qnli Text Classifier
        if self.hparams.config["loss_names"]['qnli'] > 0:
            # self.text_only = True
            self.qnli_classifier = heads.UniModalClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.qnli_classifier.apply(objectives.init_weights)
            
        # mnli Text Classifier
        if self.hparams.config["loss_names"]['mnli'] > 0:
            # self.text_only = True
            self.mnli_classifier = heads.UniModalClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=3
            )
            self.mnli_classifier.apply(objectives.init_weights)
        # cola Text Classifier
        if self.hparams.config["loss_names"]['cola'] > 0:
            # self.text_only = True
            self.cola_classifier = heads.UniModalClassificationHead(
                hidden_size=self.text_hs, 
                num_labels=2
            )
            self.cola_classifier.apply(objectives.init_weights)
        
        # if self.text_only:
        #     self.text_classification_pooler = heads.Pooler(self.text_hs)
        #     self.text_classification_pooler.apply(objectives.init_weights)
            
        
        ### Image-Only Tasks ###
        # Image-Only Classfification
        self.image_only = False
        
        if self.model_type == 'one-tower':
            self.image_hs = self.hidden_size
        else:
            self.image_hs = config['image_encoder_hidden_size']
        
        # CIFAR-10 Image Classifier
        if self.hparams.config["loss_names"]['cifar10'] > 0:
            self.image_only = True
            self.cifar10_classifier = heads.UniModalClassificationHead(
                hidden_size=self.image_hs, 
                num_labels=10
            )
            self.cifar10_classifier.apply(objectives.init_weights)
        if self.image_only:
            # Image-Only Classification Pooler
            self.image_classification_pooler = heads.Pooler(self.image_hs)
            self.image_classification_pooler.apply(objectives.init_weights)
            
        
        renaissance_utils.set_metrics(self)
        self.current_tasks = list()

        # Load Downstream (test_only = True)
        if self.test_only:
            # ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            # state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            
    def infer(self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
        image_embeds=None,
        image_masks=None,
    ):
        if self.model_type == 'one-tower':
            ret = self.encoder(
                batch,
                mask_text=mask_text,
                mask_image=mask_image,
                image_token_type_idx=image_token_type_idx,
                image_embeds=image_embeds,
                image_masks=image_masks,
            )
            return ret
        elif self.model_type == 'two-tower':
            ret = self.encoder(
                batch,
                mask_text=mask_text,
                mask_image=mask_image,
                image_token_type_idx=image_token_type_idx,
                img=img,
            )
            return ret
        return ret
    
    
    
    # Review and if update, if needed for one-tower
    def infer_text_only(self, batch):
        if self.model_type == 'two-tower':
            hidden_state = self.encoder.text_transformer(**batch).last_hidden_state
        elif self.model_type == 'one-tower':
            hidden_state = self.encoder.forward_text(batch)
        
        return hidden_state

    # This is ugly. Try to generalize
    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # SNLI Visual Entailment
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))
            
        # Reference Resolution Task
        if 'ref' in self.current_tasks:
            ret.update(objectives.compute_ref(self, batch))
        
        # Text Only Tasks
        
        # MRPC Task from GLUE
        if 'mrpc' in self.current_tasks:
            ret.update(objectives.compute_mrpc(self, batch))
        
        # rte Task from GLUE
        if 'rte' in self.current_tasks:
            ret.update(objectives.compute_rte(self, batch))
        
        # wnli Task from GLUE
        if 'wnli' in self.current_tasks:
            ret.update(objectives.compute_wnli(self, batch))
            
        # sst2 Task from GLUE
        if 'sst2' in self.current_tasks:
            ret.update(objectives.compute_sst2(self, batch))
            
        # qqp Task from GLUE
        if 'qqp' in self.current_tasks:
            ret.update(objectives.compute_qqp(self, batch))
            
        # qnli Task from GLUE
        if 'qnli' in self.current_tasks:
            ret.update(objectives.compute_qnli(self, batch))
            
        # mnli Task from GLUE
        if 'mnli' in self.current_tasks:
            ret.update(objectives.compute_mnli(self, batch))
            
        # cola Task from GLUE
        if 'cola' in self.current_tasks:
            ret.update(objectives.compute_cola(self, batch))
        
        # cifar10 Image-Only Classification Task
        if 'cifar10' in self.current_tasks:
            ret.update(objectives.compute_cifar10(self, batch))
            
        return ret

    def training_step(self, batch, batch_idx):
        renaissance_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def on_train_epoch_end(self):
        renaissance_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        renaissance_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self):
        renaissance_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        renaissance_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def on_test_epoch_end(self):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        # if self.hparams.config["loss_names"]["vqa"] > 0:
        #     objectives.vqa_test_wrapup(outs, model_name)
        renaissance_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return renaissance_utils.set_schedule(self)
