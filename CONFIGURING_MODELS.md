# Configuring Models for Pretraining, Fine-Tuning and Evaluation

Models are configured with the aid of the sacred module. All changes to model configurations are made in the renaissance/config.py file. This file explains how to use config.py for creating and training various models.

## Base Architecture

Two base architectures are available, one-tower and two-tower encoders. These model-tpyes
can be adjusted via:

```python
model_type = "two-tower" # Supports ['one-tower', 'two-tower]
```

### Configuring One-Tower Encoders

#### Choose Encoder Module
One-tower models require the user to specify a single transformer encoder from the huggingface hub. Most text transformers and some vision transformers (ViT, Beit, Deit, etc.) can serve as the encoder module.

```python
encoder = "google/electra-small-discriminator"
```

#### Choose Pooler Type
The user can choose how to pool the output of the encoder module. The 'doulbe' pooler option will concatenate the 'CLS' feature from the visual and textual inputs. The 'single' pooler option will simply take the first 'CLS' feature from both inputs.

```python
pooler_type = 'double' # Supports ['single', 'double']
```

#### Randomly Initilize Encdoer Weights
By default, Renaissance uses pretrained weights from the the encoder downloaded from the huggingface hub. However when the random_init_encoder is set to False, the encoder's values will be randomly intitialized. This is often useful when establishing baselines. Code sample below:

```python
random_init_encoder = True
```

#### Manually Configure Encoder Dimensions
By default, the encoder module will use the dimensions of the encoder downloaded from huggingface. However, if both the variables random_init_encoder and encoder_manual_configuration are set to true, the encoder will have randomly initialized weights with the provided model dimensions. Example below:

```python
random_init_encoder = True
## Manual Configuration
encoder_manual_configuration = True
hidden_size = 192
num_heads = 4
num_layers = 12
mlp_ratio = 4
drop_rate = 0.1
embedding_size = 96
```
### Configuring Two-Tower Encoders

#### Choose Text-Encoder Module
Two-tower models require the user to specify a text transformer encoder from the huggingface hub. 

```python
text_encoder = "google/electra-small-discriminator"
```

#### Choose Vision-Encoder Module
Two-tower models require the user to also specify a vision transformer encoder from the huggingface hub. A variety of vision transformer models will work, however using convolutional models that contain higher-dimensional layers, ResNet for example, will raise errors. 

```python
image_encoder = "facebook/deit-tiny-patch16-224"
```

#### Configure Cross-modal-Encoder Dimensions
Finally, two-tower models use a fusion encoder with cross-attention to combine the vision and language streams. This module is always randomly intizialized in pretraining and must always be manually configured. The settings for choosing its properties are shown below:

```python
# Cross Layer Settings
cross_layer_hidden_size = 256
num_cross_layers = 6
num_cross_layer_heads = 4
cross_layer_mlp_ratio = 4
cross_layer_drop_rate = 0.1
```
#### Randomly Initilize Text and/or Vision Encoder Weights
If the user would like to randomly initialize the weights of either encoder the text or vision encoder module, they can simply set the relevant variables to False.

```python
## Train encoder model from scratch
random_init_vision_encoder = False
# Train Text Encoder from Sratch if True
random_init_text_encoder = False
```

#### Manually Configure Text and/or Image Encoder Dimensions
The user can manually configure the dimensions of the vision encoder if both the  random_init_vision_encoder and image_encoder_manual_configuration  variables are set to True.

```python
random_init_vision_encoder = True
image_encoder_manual_configuration = True
## Manual Configuration
image_encoder_hidden_size = 192
image_encoder_num_heads = 4
image_encoder_num_layers = 12
image_encoder_mlp_ratio = 4
image_encoder_drop_rate = 0.1
image_encoder_embedding_size = 128
image_size = 224
original_image_size = 224 # Image size model is pretrained with, used in fine-tuning and testing
patch_size = 16
image_only = False
```

The same is true of the text encoder using the relevant variables:

```python
random_init_text_encoder = True
text_encoder_manual_configuration = True
text_encoder_hidden_size = 192
text_encoder_num_heads = 4
text_encoder_num_layers = 12
text_encoder_mlp_ratio = 4
text_encoder_drop_rate = 0.1
text_encoder_embedding_size = 64
max_text_len = 40
vocab_size = 30522
```

## Pretraining Configuration

There are currently two pretraining tasks available: masked language modeling (mlm) and image-text matching (itm). To select these tasks simply set their values in _loss_names to 1. The code below for example will set the model to run mlm and itm over the MSCOCO and Visual Genome datasets.

```python
datasets = ["coco", "vg"]
loss_names = _loss_names({"itm": 1, "mlm": 1})
```

Additionally, for mlm, whole word masking can be used and the probabilty that any word will be masked can be adjusted via:

```python
# Masked Language Mmodeling
whole_word_masking = True # note that whole_word_masking does not work for RoBERTa
mlm_prob = 0.15
```

For image-text matching, the number of false images and text strings to draw for each match can be adjusted with:

```python
# Image-Text Matching
draw_false_image = 1
draw_false_text = 0 
```


## Downstream Settings

## Optimizer Settings

A variety of optimization settings are available:

```python
# Optimizer Setting
optim_type = "adamw"
learning_rate = 1e-5
weight_decay = 0.01
decay_power = 1
max_epoch = 100
max_steps = 100000
warmup_steps = 10000
end_lr = 0
lr_mult_head = 5  # multiply lr for downstream heads
lr_mult_cross_modal = 5  # multiply lr for the cross-modal module
```

## General Training Settings

Finally, other general training settings are available as:

```python
seed = 0
batch_size = 256  # desired batch size; pl accumulates gradients when per_gpu_batchsize is smaller.
per_gpu_batchsize = 64  # you should define this manually with per_gpu_batch_size=#
eval_batch_size = 32
# Path to .ckpt file for fine-tuning or testing
load_path = ""
# Path to .ckpt file for resuming training from previous checkpoint
resume_from = None
# below params varies with the environment
data_root = 'data/arrow/' 
log_dir = "result"
num_gpus = 2
num_nodes = 1
num_workers = 12
precision = 32
```

## Named Configurations

In practice it is often better to group all of the necessary configuration settings into a single named configuration. This is done by organizing the desired values into a function deocorated with @ex.named_config. For example:

```python
@ex.named_config
def task_mlm_itm_deit_electra():
    exp_name = "mlm_itm_deit_electra"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 176
    per_gpu_batchsize = 44
    max_epoch = None
    max_steps = 50000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    image_only = False
    # Text Setting
    text_encoder = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = True # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Optimizer Settings
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
```

Then to run this experiment is simply:

```bash
python3 run.py with task_mlm_itm_deit_electra
```
