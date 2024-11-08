import math
import collections
import torch
import torch.nn as nn

from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.electra import ElectraConfig

from typing import List, Optional, Tuple, Union

from .objectives import init_weights
from .heads import Pooler




from transformers.models.auto import AutoConfig, AutoModel



def _get_resized_embeddings(
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc


        Return:
            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """

        if new_num_tokens is None:
            return old_embeddings
        
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        new_embeddings.apply(init_weights)

        # Copy token embeddings from the previous weights
        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings

class ElectraEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings. Based on
    the ElectraEmbeddings class from 
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/electra/modeling_electra.py"""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]


        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds 
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def _adjust_position_embeddings(self, new_num_tokens: int):
        old_text_position_embeddings = self.position_embeddings
        new_text_position_embeddings = _get_resized_embeddings(old_text_position_embeddings,new_num_tokens=new_num_tokens)
        self.position_embeddings = new_text_position_embeddings
        self.register_buffer(
            "position_ids", torch.arange(new_num_tokens).expand((1, -1)), persistent=False
        )
        
# These ViT classes still need to be adapted to renaissance program!!!
class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings

class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
        
class OneTowerEncoder(nn.Module):
    def __init__(
            self, 
            config,
            image_size,
            max_text_len,
            fine_tune,
            test_only
    ):
        super().__init__()
        
        if config['random_init_encoder']:
            # Manually Configure Encoder Dimensions
            if config['encoder_manual_configuration']:
                encoder_kwargs = {
                    'vocab_size' : config["vocab_size"],
                    'hidden_size' : config["hidden_size"],
                    'num_hidden_layers' : config["num_layers"],
                    'num_attention_heads' : config["num_heads"],
                    'intermediate_size' : config["hidden_size"] * config["mlp_ratio"],
                    'max_position_embeddings' : config["max_text_len"],
                    'hidden_dropout_prob' : config["drop_rate"],
                    'attention_probs_dropout_prob' : config["drop_rate"],
                }
                hf_config = AutoConfig.from_pretrained(config['encoder'], **encoder_kwargs)
            # Use Default Encoder Dimensions with Random Weights
            elif not config['encoder_manual_configuration']:
                hf_config = AutoConfig.from_pretrained(config['encoder'])
            model = AutoModel.from_config(hf_config)
            self.encoder = model.encoder
            
            # image_size = config['image_size']
            # max_text_len = config['max_text_len']
            # self.hidden_size = config['hidden_size']
            # self.embedding_size = config['embedding_size']
        # Use Pretrained Encoder Weights from Huggingface Hub
        else:
            # Download Encoder - Get Dimensions
            model = AutoModel.from_pretrained(config['encoder'])
            self.encoder = model.encoder
        self.hidden_size = self.encoder.config.hidden_size
        try:
            self.embedding_size = self.encoder.config.embedding_size
        except:
            self.embedding_size = self.hidden_size
            
        
        if self.embedding_size != self.hidden_size:
            self.text_embedding_projection = nn.Linear(self.embedding_size, self.hidden_size)
            self.image_embedding_projection = nn.Linear(self.embedding_size, self.hidden_size)
        
        image_config = ViTConfig(
            image_size=image_size,
            patch_size=config['patch_size'],
            hidden_size=self.embedding_size,
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        
        text_config = ElectraConfig(
            vocab_size=config["vocab_size"],
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size,
            max_position_embeddings=max_text_len,
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        
        # Add ability to adjust embedding size for down stream changes
        self.text_embeddings = ElectraEmbeddings(text_config)
        self.text_embeddings.apply(init_weights)
        
        self.image_embeddings = ViTEmbeddings(image_config)
        self.image_embeddings.apply(init_weights)
        
        self.token_type_embeddings = nn.Embedding(2, self.embedding_size)
        self.token_type_embeddings.apply(init_weights)
        
        self.pooler_type = config['pooler_type']
        if self.pooler_type == 'single':
            self.pooler = Pooler(self.hidden_size)
            self.pooler.apply(init_weights)
        elif self.pooler_type =='double':
            self.text_pooler = Pooler(self.hidden_size)
            self.text_pooler.apply(init_weights)
            self.image_pooler = Pooler(self.hidden_size)
            self.image_pooler.apply(init_weights)

    
    def forward(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
    
        text_embeds = self.text_embeddings(text_ids)
        
        image_embeds = self.image_embeddings(batch[imgkey][0], interpolate_pos_encoding=True)
        image_masks = torch.ones_like(image_embeds[:,:,0], dtype=torch.long)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                
                torch.full_like(image_masks, image_token_type_idx))
        )
        
        if self.embedding_size != self.hidden_size:
            text_embeds = self.text_embedding_projection(text_embeds)
            image_embeds = self.image_embedding_projection(image_embeds)
        
        # ERROR: Causes shape error with one-tower model.
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        x = self.encoder(x)[0]
        
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        
        if self.pooler_type == 'single':
            cls_feats = self.pooler(x)
        else:
            cls_feats_text = self.text_pooler(text_feats)
            cls_feats_image = self.image_pooler(image_feats)
            cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
            

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            'text_labels' : text_labels,
            'text_ids' : text_ids
        }

        return ret
    
    def forward_text(
        self,
        batch
    ):
    
        input_ids = batch['input_ids']
        hidden_states = self.text_embeddings(input_ids)
        if hasattr(self, "text_embedding_projection"):
            hidden_states = self.text_embedding_projection(hidden_states)
        hidden_states = self.encoder(hidden_states)[0]
        
        return hidden_states
        
    
    def get_hidden_size(self):
        return self.hidden_size
    
    def get_embedding_size(self):
        return self.embedding_size
    
    def adjust_type_embeds_for_nlvr2(self):
        emb_data = self.token_type_embeddings.weight.data
        emb_dim = self.token_type_embeddings.weight.data.shape[1]
        self.token_type_embeddings = nn.Embedding(3, emb_dim)
        self.token_type_embeddings.apply(init_weights)
        self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
        self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
        self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]