import math
import torch
import torch.nn as nn

from transformers.models.auto import AutoConfig, AutoModel
from transformers.models.lxmert.modeling_lxmert import LxmertXLayer
from transformers.models.lxmert.configuration_lxmert import LxmertConfig
from transformers.models.bert import BertConfig
from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from .objectives import init_weights
from .heads import Pooler

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        #if True:
        if False:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients) 

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = None #past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class BertCrossModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["cross_layer_hidden_size"],
            num_attention_heads=config["num_cross_layer_heads"],
            intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["cross_layer_drop_rate"],
            attention_probs_dropout_prob=config["cross_layer_drop_rate"],
        )

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_cross_layers'])])
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_cross_layers'])])

        self.cross_modal_image_pooler = Pooler(config["cross_layer_hidden_size"])
        self.cross_modal_text_pooler = Pooler(config["cross_layer_hidden_size"])
        
    def forward(self, text_embeds, image_embeds, extend_text_masks, extend_image_masks):
        x, y = text_embeds, image_embeds
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        cls_feats_image = self.cross_modal_image_pooler(y)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        
        return cls_feats, text_feats, image_feats


class LxmertCrossModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        lxmert_config = LxmertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["cross_layer_hidden_size"],
            num_attention_heads=config["num_cross_layer_heads"],
            intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["cross_layer_drop_rate"],
            attention_probs_dropout_prob=config["cross_layer_drop_rate"],
        )        

        self.cross_modal_layers = nn.ModuleList([LxmertXLayer(lxmert_config) for _ in range(config['num_cross_layers'])])
        
        self.cross_modal_image_pooler = Pooler(config["cross_layer_hidden_size"])
        self.cross_modal_text_pooler = Pooler(config["cross_layer_hidden_size"])
        
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
        


class TwoTowerEncoder(nn.Module):
    def __init__(
            self, 
            config,
            fine_tune,
            test_only
    ):
        super().__init__()
        
        self.fine_tune = fine_tune
        self.test_only = test_only
        
        self.random_init_vision_encoder = config['random_init_vision_encoder']
        self.random_init_text_encoder = config['random_init_text_encoder']
        
        # Vision Encoder
        if self.random_init_vision_encoder:
            if config['image_encoder_manual_configuration']:
                image_encoder_kwargs = {
                    'hidden_size' : config["image_encoder_hidden_size"],
                    'num_hidden_layers' : config["image_encoder_num_layers"],
                    'num_attention_heads' : config["image_encoder_num_heads"],
                    'intermediate_size' : config["image_encoder_hidden_size"] * config["image_encoder_mlp_ratio"],
                    'hidden_dropout_prob' : config["image_encoder_drop_rate"],
                    'attention_probs_dropout_prob' : config["image_encoder_drop_rate"],
                }
                hf_image_config = AutoConfig.from_pretrained(config['image_encoder'], **image_encoder_kwargs)
            # elif not config['image_encoder_manual_configuration']:
            else:
                hf_image_config = AutoConfig.from_pretrained(config['image_encoder'])
            self.image_encoder = AutoModel.from_config(hf_image_config)
            # if 'clip' in (config['image_encoder']):
            #     self.image_encoder = self.image_encoder.vision_model
        
        else:
            # hf_image_config = AutoConfig.from_pretrained(config['image_encoder'])
            self.image_encoder = AutoModel.from_pretrained(config['image_encoder'])
            
        # Freeze Parameters for self.image_encoder
        if config['freeze_image_encoder']:
            for param in self.image_encoder.parameters(self):
                param.requires_grad = False
        
        # Initialize text_encoder
        # Randomly Initialize Encoder Weights
        if self.random_init_text_encoder:
            if config['text_encoder_manual_configuration']:
                text_encoder_kwargs = {
                    'hidden_size' : config["text_encoder_hidden_size"],
                    'num_hidden_layers' : config["text_encoder_num_layers"],
                    'num_attention_heads' : config["text_encoder_num_heads"],
                    'intermediate_size' : config["text_encoder_hidden_size"] * config["text_encoder_mlp_ratio"],
                    'hidden_dropout_prob' : config["text_encoder_drop_rate"],
                    'attention_probs_dropout_prob' : config["text_encoder_drop_rate"],
                }
                hf_text_config = AutoConfig.from_pretrained(config['text_encoder'], **text_encoder_kwargs)
            # elif not config['text_encoder_manual_configuration']:
            else:
                hf_text_config = AutoConfig.from_pretrained(config['text_encoder'])
            self.text_transformer = AutoModel.from_config(hf_text_config)
        else:
            # hf_text_config = AutoConfig.from_pretrained(config['text_encoder'])
            self.text_transformer = AutoModel.from_pretrained(config['text_encoder'])
        
        # Freeze Parameters for self.text_transformer
        if config['freeze_text_encoder']:
            for param in self.text_transformer.parameters():
                param.requires_grad = False
        
        
        self.image_encoder_hidden_size = self.image_encoder.config.hidden_size
        self.text_transformer_hidden_size = self.text_transformer.config.hidden_size
        self.hidden_size = config['cross_layer_hidden_size']
        # Cross Modal Layers
        self.cross_modal_text_transform = nn.Linear(self.text_transformer_hidden_size, self.hidden_size)
        self.cross_modal_text_transform.apply(init_weights)
        self.cross_modal_image_transform = nn.Linear(self.image_encoder_hidden_size, self.hidden_size)
        self.cross_modal_image_transform.apply(init_weights)
        
        # Cross-Modal Module with LXMERT Layers
        # self.fusion_encoder = BertCrossModalEncoder(config)
        self.fusion_encoder = LxmertCrossModalEncoder(config)
        self.fusion_encoder.apply(init_weights)
        
        if config['freeze_cross_modal_layers']:
            for param in self.fusion_encoder.parameters(self):
                param.requires_grad = False
        
        # Token Type Embeddings
        self.token_type_embeddings = nn.Embedding(2, config["cross_layer_hidden_size"])
        self.token_type_embeddings.apply(init_weights)
        

        # Handle Distributed Case
        # Test this on frege when time permits
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                AutoModel.from_pretrained(config['image_encoder'])
                AutoModel.from_pretrained(config['text_encoder'])
            torch.distributed.barrier()
            
    def forward(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]
        
        # Process Text Input to Text Embeddings
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch["text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape)#, device)
        
        text_embeds = self.text_transformer(inputs_embeds=text_embeds).last_hidden_state
        text_embeds = self.cross_modal_text_transform(text_embeds)
        
        # Process Image Input to Image Embeddings
        if self.fine_tune or self.test_only:
            try:
                image_embeds = self.image_encoder(img, interpolate_pos_encoding = True)
            except:
                image_embeds = self.image_encoder(img)
        else:
            image_embeds = self.image_encoder(img)
            
        # if self.is_huggingface:
        image_embeds = image_embeds.last_hidden_state
        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size())#, device)

        # Cross-Modal Processing
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        
        # cls_feats, text_feats, image_feats = self.fusion_encoder(text_embeds, image_embeds, extend_text_masks, extend_image_masks)
        cls_feats, text_feats, image_feats = self.fusion_encoder(text_embeds, extend_text_masks, image_embeds, extend_image_masks)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }
        return ret
        
    def get_hidden_size(self):
        return self.hidden_size
    
    def adjust_type_embeds_for_nlvr2(self):
        emb_data = self.token_type_embeddings.weight.data
        self.token_type_embeddings = nn.Embedding(3, self.hidden_size)
        self.token_type_embeddings.apply(init_weights)
        self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
        self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
        self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]