"""
Fusion encoder modules using cross-attention for two-tower vision language 
encoder models. Fusion encoder modules are used in 
renaissance/modules/renaissance_module.py.
"""

import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertConfig
from transformers.models.lxmert.modeling_lxmert import LxmertXLayer
from transformers.models.lxmert.configuration_lxmert import LxmertConfig
from transformers.models.auto import AutoConfig, AutoModel
from . import heads, objectives, renaissance_utils


class LxmertCrossModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        lxmert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["cross_layer_hidden_size"],
            num_attention_heads=config["num_cross_layer_heads"],
            intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["cross_layer_drop_rate"],
            attention_probs_dropout_prob=config["cross_layer_drop_rate"],
        )        

        self.cross_modal_layers = nn.ModuleList([LxmertXLayer(lxmert_config) for _ in range(config['num_cross_layers'])])
        
        self.cross_modal_image_pooler = heads.Pooler(config["cross_layer_hidden_size"])
        self.cross_modal_text_pooler = heads.Pooler(config["cross_layer_hidden_size"])
        
    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_attention_mask,                  
        output_attentions=False,
    ):
        for layer in self.cross_modal_layers:
    
            x_outputs = layer(
                lang_feats,
                lang_attention_mask,
                visual_feats,
                visual_attention_mask   
            )
            lang_feats, visual_feats = x_outputs[:2]
            
        ### TODO!!! Add pooler to extract cls features
        cls_feats_text = self.cross_modal_text_pooler(lang_feats)
        cls_feats_image = self.cross_modal_image_pooler(visual_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        
        return cls_feats, lang_feats, visual_feats
        
