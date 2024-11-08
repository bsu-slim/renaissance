# Renaissance: A Multimodal Transformr Modeling Platform

Reanissance is a straight-forward modeling platform that allows the user to train and test a variety of vision-language model configurations with minimal programming requirements. The novel feature of this platform is that models from the Huggingface hub can be easily plugged into text and vision transformer modules. This allows users to easily test and train a huge variety of novel models with relatively little programming.   

## Model Types

Renaissance currently supports two types of encoder-only models: the one-tower encoder and the two-tower encoder. 

The one-tower encoder consists of an embedding layer, an encoder module and a output layer. The encoder module can be a drawn from number of transformer encoders available on the huggingface hub. Currently only BERT-style word-piece text embeddings and image patch embeddings are available. 

![alt text](one-tower.png)

The tow-tower encoder consists of a text-encoder, an image-encoder and a cross-modal fusion encoder
followed by an output layer. The text-encoder and the image-encoder can be drawn from a number of models available on huggingface. The fusion encoder is always manually configured and trained from scratch.

![alt text](two-tower.png)


## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Pre-trained Checkpoints

Here are the pre-trained models:



## Dataset Preparation

Dataset preperation and usage is described in DATA.md.

## Quiick Start Guide

To get the most out of this program users will need to adjust the model settings in the renaissance/config.py settings. CONFIGURING_MODELS.md provides a detailed explaination of how to use the config file. Below we have provided some simple examples using only the command line.

### Pretrain A One-Tower Model
There are two pretraining tasks available, masked language modeling (mlm) and image-text matching (itm). They can be run seperately or combined. The examples below run them together, to run them individually replace task_mlm_itm with task_mlm for masked lnaguage modeling or task itm for image_text matching. 

```bash
python run.py with task_mlm_itm encoder=<ENCODER> max_steps=<TRAINING_STEPS> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> per_gpu_batchsize=<BS_FITS_YOUR_GPU> batch_size=<BATCH_SIZE> data_root=<ARROW_ROOT>
```
Here is an example. The command below will train a one-tower model with DINO-Small as the encoder. It will train for 50k steps and will use gradient accumulation to achieve the batch size of 256.

```bash
python3 run.py with task_mlm_itm encoder=facebook/dino-vits16 max_steps=50000 num_gpus=1 num_nodes=1 per_gpu_batchsize=32 batch_size=256 data_root=data/arrow/
```



### Pretrain A Two-Tower Model
```bash
python run.py with task_mlm_itm image_encoder=<IMAGE_ENCODER> text_encoder=<TEXT_ENCODER> cross_layer_hidden_size=<CROSS_LAYER_HIDDEN_SIZE> num_cross_layers=<NUM_CROSS_LAYER> max_steps=<TRAINING_STEPS> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> per_gpu_batchsize=<BS_FITS_YOUR_GPU> batch_size=<BATCH_SIZE> data_root=<ARROW_ROOT>
```

Here is an example. The command below will train a two-tower model with DeiT-Tiny as the image encoder, ELECTRA-Small as the text-encoder, and a six layer cross-modal encoder with a hidden size of 256. It will train for 50k steps and will use gradient accumulation to achieve the batch size of 256.

```bash
python3 run.py with task_mlm_itm image_encoder=facebook/deit-tiny-patch16-224 text_encoder=google/electra-small-discriminator cross_layer_hidden_size=256 num_cross_layers=6 max_steps=50000 num_gpus=1 num_nodes=1 per_gpu_batchsize=32 batch_size=256 data_root=data/arrow/
``` 

## Finetuning and Evaluation

### NLVR2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with  task_finetune_nlvr2  load_path=<PRETRAINED_MODEL> image_encoder=<IMAGE_ENCODER> text_encoder=<TEXT_ENCODER> cross_layer_hidden_size=<CROSS_LAYER_HIDDEN_SIZE> num_cross_layers=<NUM_CROSS_LAYER>  image_size=<IMAGE_SIZE> per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> data_root=<ARROW_ROOT>
```

Here is an example:
```bash
python3 run.py with task_mlm_itm load_path=... image_encoder=facebook/deit-tiny-patch16-224 text_encoder=google/electra-small-discriminator cross_layer_hidden_size=256 image_size=288 num_cross_layers=6 per_gpu_batchsize=32 num_gpus=1 num_nodes=1 data_root=data/arrow/
```

### VQAv2

```bash
python run.py with task_finetune_vqa load_path=<PRETRAINED_MODEL> image_encoder=<IMAGE_ENCODER> text_encoder=<TEXT_ENCODER> cross_layer_hidden_size=<CROSS_LAYER_HIDDEN_SIZE> num_cross_layers=<NUM_CROSS_LAYER>  image_size=<IMAGE_SIZE> per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> data_root=<ARROW_ROOT>
```

Here is an example:
```bash
python run.py with task_finetune_vqa load_path=... image_encoder=facebook/deit-tiny-patch16-224 text_encoder=google/electra-small-discriminator cross_layer_hidden_size=256 image_size=288 num_cross_layers=6 per_gpu_batchsize=32 num_gpus=1 num_nodes=1 data_root=data/arrow/
```

### SNLI-VE

```bash
python run.py with task_finetune_snli load_path=<PRETRAINED_MODEL> image_encoder=<IMAGE_ENCODER> text_encoder=<TEXT_ENCODER> cross_layer_hidden_size=<CROSS_LAYER_HIDDEN_SIZE> num_cross_layers=<NUM_CROSS_LAYER>  image_size=<IMAGE_SIZE> per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> data_root=<ARROW_ROOT>
```

Here is an example:
```bash
python run.py with task_finetune_snli load_path=... image_encoder=facebook/deit-tiny-patch16-224 text_encoder=google/electra-small-discriminator cross_layer_hidden_size=256 image_size=288 num_cross_layers=6 per_gpu_batchsize=32 num_gpus=1 num_nodes=1 data_root=data/arrow/


## Citation

```
```

## Acknowledgements

The code is based on [ViLT](https://github.com/dandelin/ViLT) licensed under [Apache 2.0](https://github.com/dandelin/ViLT/blob/master/LICENSE) and some of the code is borrowed from [CLIP](https://github.com/openai/CLIP) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).
