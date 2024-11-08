import os
import copy
import pytorch_lightning as pl
# import os
import sys
# os.environ["NCCL_DEBUG"] = "INFO"

from renaissance.config import ex
from renaissance.modules import RenaissanceTransformer
from renaissance.datamodules.multitask_datamodule import MTDataModule

import warnings
import torch
import torch.distributed as dist



@ex.automain
def main(_config):
    
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # print(_config)
    dm = MTDataModule(_config, dist=True)

    model = RenaissanceTransformer(_config)
    
    # Create name for directory to log results
    load_path = _config['load_path']
    exp_name = f'{_config["exp_name"]}'
    seed = _config['seed']
    log_dir = _config['log_dir']
    
    print('\n\n')
    print('Running Renaissance vision-language platform with:', file=sys.stderr)
    print()
    print('Experiment Info')
    print(f'Task: {exp_name}', file=sys.stderr)
    # print('Log Dir: ', model.logger.log_dir)
    print()
    print("Model Info")
    print("Model Type: ", _config['model_type'])
    if _config['model_type'] == 'one-tower':
        print("Encoder: ", _config['encoder'])
        print("Random Init: ", _config['random_init_encoder'])
        print("Manual Config: ", _config['encoder_manual_configuration'])
        print("Image Size: ", _config['image_size'])
        print("Patch Size: ", _config['patch_size'])
        
    elif _config['model_type'] == 'two-tower':
        print("Image Encoder: ", _config['image_encoder'])
        print("Freeze Image Encoder: ", _config['freeze_image_encoder'])
        print("Image Enc Random Init: ", _config['random_init_vision_encoder'])
        print("Image Enc Manual Config: ", _config['image_encoder_manual_configuration'])
        print("Image Size: ", _config['image_size'])
        print("Patch Size: ", _config['patch_size'])
        print("Text Encoder: ", _config['text_encoder'])
        print("Freeze Text Encoder: ", _config['freeze_text_encoder'])
        print("Text Enc Random Init: ", _config['random_init_text_encoder'])
        print("Text Enc Manual Config: ", _config['text_encoder_manual_configuration'])
        print("Max Text Langth: ", _config['max_text_len'])
        print("Vocab Size: ", _config['vocab_size'])
        
    print()
    print("Training Info")
    print("Learning Rate: ", _config['learning_rate'])
    print("Max Epochs: ", _config['max_epoch'])
    print("Max Steps: ", _config['max_steps'])
    print("Warmup Steps: ", _config['warmup_steps'])
    print("LR Mult Head: ", _config['lr_mult_head'])  
    print("LR Mult Cross Modal: ", _config['lr_mult_cross_modal'])
    print('\n\n')
    
    def parse_load_path(load_path):
        drive, path_and_file = os.path.splitdrive(load_path)
        path, file = os.path.split(path_and_file)
        folders = []
        while True:
            path, folder = os.path.split(path)
        
            if folder != "":
                folders.append(folder)
            else:
                if path != "":
                    folders.append(path)
                break
        folders.reverse()
        result_dir = folders[-3]
        checkpoint_name = file.split("/")[-1][:-5]
        parsed_string = f"{result_dir}_{checkpoint_name}"
        return parsed_string
        
        
    if not load_path:
        image_size = _config['image_size']
        patch_size = _config['patch_size']
        batch_size = _config['batch_size']
        per_gpu_batchsize = _config['per_gpu_batchsize']
        train_steps = _config['max_steps']
        train_epoch = _config['max_epoch']
        result_dir = f"{exp_name}_seed{seed}_is{image_size}_ps{patch_size}_bs{batch_size}_pgbs{per_gpu_batchsize}_ts{train_steps}"
    else:
        loaded_model = parse_load_path(load_path)
        result_dir = f"{exp_name}_seed{seed}_from_{loaded_model}"
        
    # Info Variables
    exp_name = _config['exp_name']
    
    
    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=result_dir
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    torch.set_float32_matmul_precision('medium')
    
    
    trainer = pl.Trainer(
        devices= _config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator = 'gpu',
        strategy='ddp_find_unused_parameters_true',
        # strategy='ddp',
        deterministic='warn',
        max_epochs=_config["max_epoch"], #if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        if _config["resume_from"]:
            trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        else:
            trainer.fit(model, datamodule=dm)
        
        # Display location of results
        print()
        print('Results can be found in:')
        print(model.logger.log_dir)
        print()
        
    else:
        trainer.test(model, datamodule=dm)
    
    
