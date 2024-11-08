"""
Classification heads for vision language 
encoder models. Classification modules are used in 
renaissance/modules/renaissance_module.py.
"""
import torch
import torch.nn as nn

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from typing import List, Optional, Tuple, Union


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


# this needs fixing to accomodate one or tow tower model
class MLMHead(nn.Module):
    def __init__(self, config, hidden_size=None, weight=None):
        
        
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=hidden_size,
            num_attention_heads=config["num_cross_layer_heads"],
            intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["cross_layer_drop_rate"],
            attention_probs_dropout_prob=config["cross_layer_drop_rate"],
        )
        super().__init__()
        self.transform = BertPredictionHeadTransform(bert_config)
        self.decoder = nn.Linear(bert_config.hidden_size, bert_config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(bert_config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

# class MultiModalClassificationHead(nn.Module):
    
#     def __init__(self,  hidden_size = None, num_labels = None):
#         self.dense = nn.Linear(hs, hs)
#         self.layer_norm = nn.LayerNorm(hs)
#         self.activation = nn.GELU()
#         self.out_proj = nn.Linear(hs, 3)

class UniModalClassificationHead(nn.Module):
    """Head for unimodal text or image classification tasks."""


    def __init__(self, hidden_size = None, num_labels = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0]  # take <s> token (equiv. to [CLS])
        # x = features
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.out_proj(x)
        return x
    
class MultiModalClassificationHead(nn.Module):
    """Head for multimodal classification tasks."""


    def __init__(self, hidden_size = None, num_labels = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0]  # take <s> token (equiv. to [CLS])
        x = features
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.out_proj(x)
        return x
    
class NLVR2ClassificationHead(nn.Module):
    """Head for NLVR2 classification tasks."""


    def __init__(self, hidden_size = None, num_labels = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dense = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, features, **kwargs):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.out_proj(x)
        return x
    
        