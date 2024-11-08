from sacred import Experiment

ex = Experiment("renaissance")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 0,
        "ref": 0,
        "mrpc" : 0,
        "rte" : 0,
        'wnli' : 0,
        'sst2' : 0,
        'qqp' : 0,
        'qnli' : 0,
        'mnli' : 0,
        'cola' : 0,
        'cifar10' : 0
    }
    ret.update(d)
    return ret

# ===================== Default Settings ===================== #
@ex.config
def config():
    exp_name = "renaissance"
    seed = 0
    datasets = ["coco", "vg"] # Supports ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    eval_batch_size = 32
    
    # Path to .ckpt file for fine-tuning or testing
    load_path = ""
    # Path to .ckpt file for resuming training from previous checkpoint
    resume_from = None

    # Model Type Setting
    model_type = "two-tower" # Supports ['one-tower', 'two-tower]
    
    #### One Tower Settings ####
    # one-tower settings will be ignored if unless model_type = "one-tower"
    # Text Setting
    encoder = "google/electra-small-discriminator"
    pooler_type = 'double' # Supports ['single', 'double']
    tokenizer = "bert-base-uncased"

    # Transformer Setting
    # Train encoder model from scratch
    random_init_encoder = False
    ## Manual Configuration
    encoder_manual_configuration = False
    hidden_size = 192
    num_heads = 4
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    embedding_size = 96

    #### Two Tower Settings ####
    ### Image Encoder settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    ## Train encoder model from scratch
    random_init_vision_encoder = False
    ## Manual Configure Image Encoder
    image_encoder_manual_configuration = False
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
    # Image Transform Keys
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]

    
    # Text Setting
    text_encoder = "google/electra-small-discriminator"
    # Train Text Encoder from Sratch if True
    random_init_text_encoder = False
    # Manual Text Settings - Ignored unless random_init_text_encoder = True
    text_encoder_manual_configuration = False
    text_encoder_hidden_size = 192
    text_encoder_num_heads = 4
    text_encoder_num_layers = 12
    text_encoder_mlp_ratio = 4
    text_encoder_drop_rate = 0.1
    text_encoder_embedding_size = 64
    max_text_len = 40
    vocab_size = 30522

    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
    # Freeze Module Parameter Settings
    freeze_image_encoder = False
    freeze_text_encoder = False
    freeze_cross_modal_layers = False   
    
    # Pretraining Settings
    # Masked Language Mmodeling
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    # Image-Text Matching
    draw_false_image = 1
    draw_false_text = 0 

    # Downstream Settings
    # Image-Text Recall
    get_recall_metric = False
    # Visual Question Answering
    vqav2_label_size = 3129

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
    
    
    # PL Trainer Setting
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = 'data/arrow/' 
    log_dir = "result"
    num_gpus = 2
    num_nodes = 1
    num_workers = 12
    precision = 32
    

# ===================== Experiment 2 Cofigs ===================== #

@ex.named_config
def pretrain_mlm_itm_twotower_exp2_deittiny_electrasmall():
    exp_name = "mlm_itm_twotower_exp2_deittiny_electrasmall"
    model_type = "two-tower"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 512
    per_gpu_batchsize = 128
    max_epoch = None
    max_steps = 50000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = "facebook/deit-tiny-distilled-patch16-224"
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
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5

@ex.named_config
def pretrain_mlm_itm_twotower_exp2_deitsmall_electrasmall():
    exp_name = "mlm_itm_twotower_exp2_deitsmall_electrasmall"
    model_type = "two-tower"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 512
    per_gpu_batchsize = 128
    max_epoch = None
    max_steps = 50000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = "facebook/deit-small-distilled-patch16-224"
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
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
@ex.named_config
def pretrain_mlm_itm_twotower_exp2_deittiny_electratiny():
    exp_name = "mlm_itm_exp2_deittiny_electratiny"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 512
    per_gpu_batchsize = 128
    max_epoch = None
    max_steps = 50000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = "facebook/deit-tiny-distilled-patch16-224"
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    # Text Setting
    text_encoder = "claytonfields/electra-tiny"
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
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
@ex.named_config
def pretrain_mlm_itm_twotower_exp2_deitsmall_electratiny():
    exp_name = "mlm_itm_exp2_deitsmall_electratiny"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 512
    per_gpu_batchsize = 128
    max_epoch = None
    max_steps = 50000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = "facebook/deit-small-distilled-patch16-224"
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    # Text Setting
    text_encoder = "claytonfields/electra-tiny"
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
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
@ex.named_config
def pretrain_mlm_itm_twotower_exp2_swintiny_electratiny():
    exp_name = "mlm_itm_exp2_swintiny_electratiny"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 512
    per_gpu_batchsize = 128
    max_epoch = None
    max_steps = 50000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = "facebook/deit-small-distilled-patch16-224"
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    # Text Setting
    text_encoder = "claytonfields/electra-tiny"
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
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
@ex.named_config
def pretrain_mlm_itm_twotower_exp2_swintiny_electrasmall():
    exp_name = "mlm_itm_exp2_swintiny_electrasmall"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 512
    per_gpu_batchsize = 128
    max_epoch = None
    max_steps = 50000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = "facebook/deit-small-distilled-patch16-224"
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    # Text Setting
    text_encoder = "claytonfields/electra-tiny"
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
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
@ex.named_config
def finetune_nlvr2_twotower_exp2_deittiny_electrasmall():
    exp_name = "nlvr2_twotower_exp2_deittiny_electrasmall"
    model_type = "two-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Hardware Settings
    per_gpu_batchsize = 32 
    num_nodes = 1 
    num_gpus = 2 
    # Training Settings
    batch_size = 256
    max_epoch = 40  
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5  
    lr_mult_cross_modal = 5 
    # Text Setting
    text_encoder = "google/electra-small-discriminator"
    max_text_len = 50
    # Image Settings
    image_encoder = "facebook/deit-tiny-distilled-patch16-224"
    patch_size = 16
    image_size = 288
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

@ex.named_config
def finetune_nlvr2_twotower_exp2_deittiny_electratiny():
    exp_name = "nlvr2_twotower_exp2_deittiny_electratiny"
    model_type = "two-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Hardware Settings
    per_gpu_batchsize = 32 
    num_nodes = 1 
    num_gpus = 2 
    # Training Settings
    batch_size = 256
    max_epoch = 40  
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5  
    lr_mult_cross_modal = 5 
    # Text Setting
    text_encoder = "claytonfields/electra-tiny"
    max_text_len = 50
    # Image Settings
    image_encoder = "facebook/deit-tiny-distilled-patch16-224"
    patch_size = 16
    image_size = 288
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1


    
@ex.named_config
def finetune_snli_twotower_exp2_deittiny_electrasmall():
    exp_name = "snli_twotower_exp2_deittiny_electrasmall"
    model_type = "two-tower"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Hardware Settings
    per_gpu_batchsize = 64  
    num_nodes = 1 
    num_gpus = 2
    # Training Settings
    batch_size = 64
    max_epoch = 10 
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False

    # Encoder Settings
    text_encoder = "google/electra-small-discriminator"
    image_encoder = "facebook/deit-tiny-distilled-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
@ex.named_config
def finetune_snli_twotower_exp2_deittiny_electratiny():
    exp_name = "snli_twotower_exp2_deittiny_electratiny"
    model_type = "two-tower"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Hardware Settings
    per_gpu_batchsize = 64  
    num_nodes = 1 
    num_gpus = 2
    # Training Settings
    batch_size = 64
    max_epoch = 10 
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False

    # Encoder Settings
    text_encoder = "claytonfields/electra-tiny"
    image_encoder = "facebook/deit-tiny-distilled-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

@ex.named_config
def finetune_ref_twotower_exp2_deittiny_electrasmall():
    exp_name = "ref_twotower_exp2_deittiny_electrasmall"
    model_type = "two-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    max_text_len = 40
    image_size = 224
    
    # Encoder Settings
    text_encoder = "google/electra-small-discriminator"
    image_encoder = "facebook/deit-tiny-distilled-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
@ex.named_config
def finetune_ref_twotower_exp2_deittiny_electratiny():
    exp_name = "ref_twotower_exp2_deittiny_electratiny"
    model_type = "two-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    max_text_len = 40
    image_size = 224
    
    # Encoder Settings
    text_encoder = "claytonfields/electra-tiny"
    image_encoder = "facebook/deit-tiny-distilled-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

# ===================== Pretraining Tasks===================== #

@ex.named_config
def demo():
    exp_name = "demo"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 176
    per_gpu_batchsize = 44
    num_nodes = 1
    num_gpus = 2
    max_epoch = None
    max_steps = 200
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
    # resolution_before = 224
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


@ex.named_config
def pretrain_mlm():
    exp_name = "mlm"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({ "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    model_type = "two-tower"

    max_text_len = 50
    image_size = 224
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
@ex.named_config
def pretrain_mlm_onetower():
    exp_name = "mlm_onetower"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({ "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    model_type = "two-tower"

    max_text_len = 50
    image_size = 224
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5


@ex.named_config
def pretrain_itm_twotower():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    
    model_type = "two-tower"
    
    draw_false_image = 1
    max_text_len = 50
    image_size = 224
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5

@ex.named_config
def pretrain_mlm_itm():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    draw_false_image = 1
    max_text_len = 50
    image_size = 224
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
@ex.named_config
def pretrain_mlm_itm_onetower():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    draw_false_image = 1
    max_text_len = 50
    image_size = 224
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
    # Transformer Setting
    model_type = "one-tower"
    
@ex.named_config
def pretrain_onetower_mlm_itm_manual_config_electra_exp1():
    exp_name = "test_onetower_mlm_itm__manual_config_electra_exp1"
    seed = 0
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 704
    per_gpu_batchsize = 176
    num_gpus=2
    num_nodes=1 
    
    max_epoch = None
    max_steps = 100000
    # Optimizer Settings
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    warmup_steps = 0.1
    
    # Transformer Setting
    model_type = "one-tower"
    encoder = "google/electra-base-discriminator"
    # Train encoder model from scratch
    random_init_encoder = True
    ## Manual Configuration
    encoder_manual_configuration = True
    hidden_size = 448
    num_heads = 4
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    embedding_size = 128
    
    # Image setting
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    
    # Text Setting
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = True
    mlm_prob = 0.15
    draw_false_text = 0
    
    
    
@ex.named_config
def pretrain_mlm_itm_twotower():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    model_type = "two-tower"
    
    draw_false_image = 1
    max_text_len = 50
    image_size = 224
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5

@ex.named_config
def pretrain_mlm_itm_onetower_deitsmall():
    exp_name = "mlm_itm_onetower_deitsmall"
    seed = 0
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 704  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    per_gpu_batchsize = 176
    
    # Transformer Setting
    model_type = "one-tower"
    encoder = "facebook/deit-small-distilled-patch16-224"
    
    # Image setting
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    image_only = False
    
    # Text Setting
    max_text_len = 40
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0
    
    # Optimizer Settings
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    
    max_steps = 100000


    
@ex.named_config
def pretrain_mlm_itm_onetower_electra():
    exp_name = "mlm_itm_onetower_electra_small"
    seed = 0
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 44  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    per_gpu_batchsize = 44
    
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=44
    resume_from = None
    
    # Transformer Setting
    model_type = "one-tower"
    encoder = "google/electra-small-discriminator"
    
    # Image setting
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    image_only = False
    
    # Text Setting
    max_text_len = 40
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0
    
    max_steps = 50000
    
@ex.named_config
def pretrain_mlm_itm_onetower_electra_base():
    exp_name = "mlm_itm_onetower_electra_base"
    seed = 0
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 176  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    per_gpu_batchsize = 44
    
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=44
    resume_from = None
    
    # Transformer Setting
    model_type = "one-tower"
    encoder = "google/electra-base-discriminator"
    drop_rate = 0.1
    
    # Image setting
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    image_only = False
    draw_false_image = 1
    
    # Text Setting
    vocab_size = 30522
    whole_word_masking = True
    mlm_prob = 0.15
    
    max_steps = 50000
    max_epoch = None
    
@ex.named_config
def pretrain_mlm_itm_onetower_bert_base():
    exp_name = "mlm_itm_onetower_bert_base"
    seed = 0
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 1024  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    per_gpu_batchsize = 128
    
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=44
    resume_from = None
    
    # Transformer Setting
    model_type = "one-tower"
    encoder = "google-bert/bert-base-uncased"
    drop_rate = 0.1
    
    # Image setting
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    image_only = False
    draw_false_image = 1
    
    # Text Setting
    vocab_size = 30522
    whole_word_masking = True
    mlm_prob = 0.15
    
    max_steps = 50000
    max_epoch = None
    

@ex.named_config
def pretrain_mlm_itm_onetower_dinos16():
    exp_name = "mlm_itm_onetower_dinos16"
    seed = 0
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 704  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    per_gpu_batchsize = 176
    num_gpus=2 
    num_nodes=1 
    # Transformer Setting
    model_type = "one-tower"
    encoder = "facebook/dino-vits16"
    
    # Image setting
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    image_only = False
    
    # Text Setting
    max_text_len = 50
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = True
    mlm_prob = 0.15
    draw_false_text = 0
    
    max_steps = 100000
    
@ex.named_config
def pretrain_mlm_itm_onetower_swinsmall():
    exp_name = "mlm_itm_onetower_swinsmall"
    seed = 0
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 704  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    per_gpu_batchsize = 176
    num_gpus=2 
    num_nodes=1 
    # Transformer Setting
    model_type = "one-tower"
    encoder = "microsoft/swin-small-patch4-window7-224"
    
    # Image setting
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    
    # Text Setting
    max_text_len = 50
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = True
    mlm_prob = 0.15
    draw_false_text = 0
    
    max_steps = 100000
    

@ex.named_config
def pretrain_mlm_itm_onetower_vit_base():
    exp_name = "mlm_itm_onetower_vit_base"
    seed = 0
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 176  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    per_gpu_batchsize = 44
    
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=44
    resume_from = None
    
    # Transformer Setting
    model_type = "one-tower"
    encoder = "google/vit-base-patch16-224"
    drop_rate = 0.1
    
    # Image setting
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    image_only = False
    draw_false_image = 1
    
    # Text Setting
    vocab_size = 30522
    whole_word_masking = True
    mlm_prob = 0.15
    
    max_steps = 50000
    max_epoch = None
    
    
@ex.named_config
def pretrain_mlm_itm_twotower_deit_electra():
    exp_name = "mlm_itm_deit_electra"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
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
    # resolution_before = 224
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

@ex.named_config
def pretrain_mlm_itm_twotower_deit_electra_exp1():
    exp_name = "mlm_itm_deit_electra_exp1"
    model_type = "two-tower"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 704
    per_gpu_batchsize = 176
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = "facebook/deit-tiny-patch16-224"
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
    cross_layer_hidden_size = 292
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Optimizer Settings
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    


    
@ex.named_config
def pretrain_mlm_itm_twotower_deittiny_electratiny():
    exp_name = "mlm_itm_deittiny_electratiny"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 704
    per_gpu_batchsize = 176
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    # Text Setting
    text_encoder = "claytonfields/electra-tiny"
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
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    

    
@ex.named_config
def pretrain_mlm_itm_twotower_electrasmall_swintiny():
    exp_name = "mlm_itm_electrasmall_swintiny"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 704
    per_gpu_batchsize = 176
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    # Text Setting
    text_encoder = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = True # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 7
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Optimizer Settings
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    


@ex.named_config
def pretrain_mlm_itm_twotower_deit_fr_electra():
    exp_name = "mlm_itm_deit_fr_electra"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    # Freeze Image Encoder
    freeze_image_encoder = True
    # DO NOT Freeze Text Encoder
    freeze_text_encoder = False
    
    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    draw_false_image = 1
    
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    num_cross_layers = 6
    
@ex.named_config
def pretrain_mlm_itm_twotower_deit_electra_fr():
    exp_name = "mlm_itm_deit_electra_fr"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    # DO NOT Freeze Image Encoder
    freeze_image_encoder = False
    # Freeze Text Encoder
    freeze_text_encoder = True
    
    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    draw_false_image = 1
    
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    num_cross_layers = 6

@ex.named_config
def pretrain_mlm_itm_twotower_deit_fr_electra_fr():
    exp_name = "mlm_itm_deit_fr_electra_fr"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    # Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True
    
    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    draw_false_image = 1
    
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    num_cross_layers = 6
    
@ex.named_config
def pretrain_mlm_itm_two_tower_dinos_tinybert():
    exp_name = "mlm_itm_dinos_tinybert"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    draw_false_image = 1
    
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
    # Encoders
    num_cross_layers = 6
    cross_layer_hidden_size = 256
    
    image_encoder = "facebook/dino-vits16"
    text_encoder = "huawei-noah/TinyBERT_General_4L_312D"
    
    # Image setting
    image_size = 224
    patch_size = 16
    
@ex.named_config
def pretrain_mlm_itm_two_tower_dinos_rnd_tinybert():
    exp_name = "mlm_itm_dinos_rnd_tinybert"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    draw_false_image = 1
    
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
    # Encoders
    num_cross_layers = 6
    cross_layer_hidden_size = 256
    
    image_encoder = "facebook/dino-vits16"
    text_encoder = "huawei-noah/TinyBERT_General_4L_312D"
    
    ## Train Vision encoder model from scratch
    random_init_vision_encoder = True
    
    # Image setting
    image_size = 224
    patch_size = 16
    
@ex.named_config
def pretrain_mlm_itm_two_tower_dinos_tinybert_rnd():
    exp_name = "mlm_itm_dinos_tinybert_rnd"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    draw_false_image = 1
    
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
    # Encoders
    num_cross_layers = 6
    cross_layer_hidden_size = 256
    
    image_encoder = "facebook/dino-vits16"
    text_encoder = "huawei-noah/TinyBERT_General_4L_312D"
    
    # Train Text Encoder from Sratch if True
    random_init_text_encoder = True
    
    # Image setting
    image_size = 224
    patch_size = 16
    
@ex.named_config
def pretrain_mlm_itm_two_tower_dinos_rnd_tinybert_rnd():
    exp_name = "mlm_itm_dinos_rnd_tinybert_rnd"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    draw_false_image = 1
    
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
    # Encoders
    num_cross_layers = 6
    cross_layer_hidden_size = 256
    
    image_encoder = "facebook/dino-vits16"
    text_encoder = "huawei-noah/TinyBERT_General_4L_312D"
    
    # Train Vision and Text Encoder from Sratch if True
    random_init_vision_encoder = True
    random_init_text_encoder = True
    
    # Image setting
    image_size = 224
    patch_size = 16
    
    
@ex.named_config
def pretrain_mlm_itm_twotower_vitbase_fr_electrabase_fr():
    exp_name = "mlm_itm_twotower_vitbase_fr_electrabase_fr"
    model_type = "two-tower"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 704
    per_gpu_batchsize = 176
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True
    # Image settings
    image_encoder = "google/vit-base-patch16-224"
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_size = 224
    # resolution_before = 224
    patch_size = 16
    draw_false_image = 1
    image_only = False
    # Text Setting
    text_encoder = "google/electra-base-discriminator"
    max_text_len = 50
    whole_word_masking = True # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 10
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Optimizer Settings
    learning_rate = 7.5e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    
    
# ===================== Finetuning Tasks===================== #
@ex.named_config
def finetune_nlvr2():
    exp_name = "nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Training Settings
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    # Image Size
    image_size = 288
    
@ex.named_config
def finetune_nlvr2_onetower():
    exp_name = "nlvr2_onetower"
    model_type = "one-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Hardware Settings
    per_gpu_batchsize = 32 
    num_nodes = 1 
    num_gpus = 2 
    # Training Settings
    batch_size = 256
    max_epoch = 40  
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5  
    # Text Setting
    max_text_len = 50
    # Image Settings
    patch_size = 16
    image_size = 288

@ex.named_config
def finetune_nlvr2_onetower_deitsmall():
    exp_name = "nlvr2_onetower_deitsmall"
    model_type = "one-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Hardware Settings
    per_gpu_batchsize = 32 
    num_nodes = 1 
    num_gpus = 2 
    # Training Settings
    batch_size = 256
    max_epoch = 40  
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5  
    # Text Setting
    max_text_len = 50
    # Image Settings
    patch_size = 16
    image_size = 288
    # Encoder Settings
    encoder = "facebook/deit-small-distilled-patch16-224"


@ex.named_config
def finetune_nlvr2_onetower_manual_config_electrabase_exp1():
    exp_name = "nlvr2_onetower__manual_config_electrabase_exp1"
    model_type = "one-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Hardware Settings
    per_gpu_batchsize = 32 
    num_nodes = 1 
    num_gpus = 2 
    # Training Settings
    batch_size = 256
    max_epoch = 40  
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5  
    # Text Setting
    max_text_len = 50
    # Image Settings
    patch_size = 16
    image_size = 288
    # Encoder Settings
    model_type = "one-tower"
    encoder = "google/electra-base-discriminator"
    # Train encoder model from scratch
    random_init_encoder = True
    ## Manual Configuration
    encoder_manual_configuration = True
    hidden_size = 448
    num_heads = 4
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    embedding_size = 128

@ex.named_config
def finetune_nlvr2_onetower_vit():
    exp_name = "nlvr2_onetower_vit"
    model_type = "one-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Training Settings
    batch_size = 256
    per_gpu_batchsize = 32
    max_epoch = 40
    max_steps = 10e6
    draw_false_image = 0
    # Image Size
    image_size = 288
    
    encoder = "google/vit-base-patch16-224"
    
    learning_rate = 7.5e-5 
    lr_mult_head = 5
    warmup_steps = 0.05
    
@ex.named_config
def finetune_nlvr2_onetower_bert():
    exp_name = "nlvr2_onetower_bert"
    model_type = "one-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Training Settings
    batch_size = 256
    per_gpu_batchsize = 32
    max_epoch = 40
    max_steps = 10e6
    draw_false_image = 0
    # Image Size
    image_size = 288
    
    encoder = "google-bert/bert-base-uncased"
    
    learning_rate = 7.5e-5 
    lr_mult_head = 5 
    warmup_steps = 0.05

@ex.named_config
def finetune_nlvr2_twotower():
    exp_name = "nlvr2_twotower"
    model_type = "two-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Training Settings
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    # Image Size
    image_size = 288
    


@ex.named_config
def finetune_nlvr2_twotower_deit_electra():
    exp_name = "nlvr2_twotower_deit_electra"
    model_type = "two-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Hardware Settings
    per_gpu_batchsize = 32 
    num_nodes = 1 
    num_gpus = 2 
    # Training Settings
    batch_size = 256
    max_epoch = 40  
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-5 
    lr_mult_head = 5  
    lr_mult_cross_modal = 15 
    # Text Setting
    text_encoder = "google/electra-small-discriminator"
    max_text_len = 50
    # Image Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    patch_size = 16
    image_size = 288
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
@ex.named_config
def finetune_nlvr2_twotower_deit_electra_exp2():
    exp_name = "nlvr2_twotower_deit_electra_exp2"
    model_type = "two-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Hardware Settings
    per_gpu_batchsize = 32 
    num_nodes = 1 
    num_gpus = 2 
    # Training Settings
    batch_size = 256
    max_epoch = 40  
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5  
    lr_mult_cross_modal = 15 
    # Text Setting
    text_encoder = "google/electra-small-discriminator"
    max_text_len = 50
    # Image Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    patch_size = 16
    image_size = 288
    # Cross Layer Settings
    cross_layer_hidden_size = 292
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    

@ex.named_config
def finetune_nlvr2_twotower_electrasmall_swintiny():
    exp_name = "nlvr2_twotower_electrasmall_swintiny"
    model_type = "two-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Hardware Settings
    per_gpu_batchsize = 32 
    num_nodes = 1 
    num_gpus = 2 
    # Training Settings
    batch_size = 256
    max_epoch = 40  
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5  
    lr_mult_cross_modal = 15 
    # Text Setting
    text_encoder = "google/electra-small-discriminator"
    max_text_len = 50
    # Image Settings
    image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    patch_size = 16
    image_size = 288
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 7
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
@ex.named_config
def finetune_nlvr2_twotower_deittiny_electratiny():
    exp_name = "nlvr2_twotower_deittiny_electratiny"
    model_type = "two-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Training Settings
    batch_size = 256
    max_epoch = 40  
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-5 
    lr_mult_head = 5  
    lr_mult_cross_modal = 15 
    # Text Setting
    text_encoder = "claytonfields/electra-tiny"
    max_text_len = 50
    # Image Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    patch_size = 16
    image_size = 288
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    
@ex.named_config
def finetune_nlvr2_twotower_vitbase_electrabase():
    exp_name = "nlvr2_twotower_vitbase_electrabase"
    model_type = "two-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Training Settings
    batch_size = 256
    max_epoch = 40  
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-5 
    lr_mult_head = 5  
    lr_mult_cross_modal = 15 
    # Text Setting
    text_encoder = "google/electra-base-discriminator"
    max_text_len = 50
    # Image Settings
    image_encoder = "google/vit-base-patch16-224"
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    patch_size = 16
    image_size = 288
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
@ex.named_config
def finetune_nlvr2_twotower_dino_tinybert():
    exp_name = "nlvr2_twotower_dinos_tinybert"
    model_type = "two-tower"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Training Settings
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    # Text Setting
    text_encoder = "huawei-noah/TinyBERT_General_4L_312D"
    max_text_len = 50
    # Image Settings
    image_encoder = "facebook/dino-vits16"
    patch_size = 16
    image_size = 288
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
    
    

@ex.named_config
def finetune_vqa_twotower():
    exp_name = "vqa_twotower"
    model_type = "two-tower"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = 1e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 0.1
    lr_mult_head = 50
    lr_mult_cross_modal = 5
    # Text Settings
    max_text_len = 50
    # Image Settins
    image_size = 576

@ex.named_config
def finetune_vqa_twotower_deit_electra():
    exp_name = "vqa_twotower_deit_electra"
    model_type = "two-tower"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = 1e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 0.1
    lr_mult_head = 50
    lr_mult_cross_modal = 5
    # Text Settings
    text_encoder = "google/electra-small-discriminator"
    max_text_len = 50
    # Image Settins
    image_encoder = "facebook/deit-tiny-patch16-224"
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_size = 576
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
@ex.named_config
def finetune_vqa_twotower_dinos_tinybert():
    exp_name = "vqa_twotower_dinos_tinybert"
    model_type = "two-tower"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = 1e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 0.1
    lr_mult_head = 50
    lr_mult_cross_modal = 5
    # Text Settings
    image_encoder = "facebook/dino-vits16"
    max_text_len = 50
    # Image Settins
    image_encoder = "facebook/deit-tiny-patch16-224"
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_size = 576
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

@ex.named_config
def finetune_snli_twotower():
    exp_name = "snli_twotower"
    model_type = "two-tower"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    max_epoch = 5
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False

@ex.named_config
def finetune_snli_twotower_dino_tinybert():
    exp_name = "snli_twotower_dino_tinybert"
    model_type = "two-tower"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Hardware Settings
    per_gpu_batchsize = 64  
    num_nodes = 1 
    num_gpus = 2
    # Training Settings
    batch_size = 64
    max_epoch = 10 
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-5  
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False

    # Encoder Settings
    text_encoder = "huawei-noah/TinyBERT_General_4L_312D"
    image_encoder = "facebook/dino-vits16"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
@ex.named_config
def finetune_snli_twotower_deit_electra():
    exp_name = "snli_twotower_deit_electra"
    model_type = "two-tower"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Hardware Settings
    per_gpu_batchsize = 64  
    num_nodes = 1 
    num_gpus = 2
    # Training Settings
    batch_size = 64
    max_epoch = 10 
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-5  
    lr_mult_head = 5
    lr_mult_cross_modal = 15
    max_text_len = 50
    image_size = 384
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False

    # Encoder Settings
    text_encoder = "google/electra-small-discriminator"
    image_encoder = "facebook/deit-tiny-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    


@ex.named_config
def finetune_snli_twotower_deit_electra_exp2():
    exp_name = "snli_twotower_deit_electra_exp2"
    model_type = "two-tower"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Hardware Settings
    per_gpu_batchsize = 64  
    num_nodes = 1 
    num_gpus = 2
    # Training Settings
    batch_size = 64
    max_epoch = 10 
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 15
    max_text_len = 50
    image_size = 384
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False

    # Encoder Settings
    text_encoder = "google/electra-small-discriminator"
    image_encoder = "facebook/deit-tiny-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 292
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

@ex.named_config
def snli_twotower_electrasmall_swintiny():
    exp_name = "snli_twotower_electrasmall_swintiny"
    model_type = "two-tower"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Hardware Settings
    per_gpu_batchsize = 64  
    num_nodes = 1 
    num_gpus = 2
    # Training Settings
    batch_size = 64
    max_epoch = 10 
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 15
    max_text_len = 50
    image_size = 384
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False

    # Encoder Settings
    text_encoder = "google/electra-small-discriminator"
    image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 7
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

@ex.named_config
def finetune_snli_twotower_deittiny_electratiny():
    exp_name = "snli_twotower_deit_electra"
    model_type = "two-tower"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Training Settings
    batch_size = 64
    max_epoch = 10 
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-5  
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False

    # Encoder Settings
    text_encoder = "claytonfields/electra-tiny"
    image_encoder = "facebook/deit-tiny-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

@ex.named_config
def finetune_snli_twotower_vitbase_electrabase():
    exp_name = "snli_twotower_deit_electra"
    model_type = "two-tower"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    max_epoch = 5
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False

    # Encoder Settings
    text_encoder = "google/electra-base-discriminator"
    image_encoder = "google/vit-base-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
@ex.named_config
def finetune_snli_onetower():
    exp_name = "snli_onetower"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Hardware Settings
    per_gpu_batchsize = 64  
    num_nodes = 1 
    num_gpus = 2
    # Training Settings
    batch_size = 64
    max_epoch = 10 
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 15
    max_text_len = 50
    image_size = 384
    patch_size = 16
    # One-tower settings
    model_type = "one-tower"
    # encoder_type = 'text'
    pooler_type = 'double' # 'double' or 'single'
    random_init_encoder = False
    
@ex.named_config
def finetune_snli_onetower_deitsmall():
    exp_name = "snli_onetower_deitsmall"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Hardware Settings
    per_gpu_batchsize = 64  
    num_nodes = 1 
    num_gpus = 2
    # Training Settings
    batch_size = 64
    max_epoch = 10 
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 15
    max_text_len = 50
    image_size = 384
    patch_size = 16
    # One-tower settings
    encoder = "facebook/deit-small-distilled-patch16-224"
    model_type = "one-tower"
    pooler_type = 'double' # 'double' or 'single'
    random_init_encoder = False
    

@ex.named_config
def finetune_snli_onetower_manual_config_electrabase_exp1():
    exp_name = "snli_onetower_manual_config_electrabase_exp1"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Hardware Settings
    per_gpu_batchsize = 64  
    num_nodes = 1 
    num_gpus = 2
    # Training Settings
    batch_size = 64
    max_epoch = 10 
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 1e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 15
    max_text_len = 50
    image_size = 384
    patch_size = 16
    # Encoder Settings
    model_type = "one-tower"
    encoder = "google/electra-base-discriminator"
    # Train encoder model from scratch
    random_init_encoder = True
    ## Manual Configuration
    encoder_manual_configuration = True
    hidden_size = 448
    num_heads = 4
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    embedding_size = 128
    
@ex.named_config
def finetune_snli_onetower_electra():
    exp_name = "snli_onetower_electra"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    per_gpu_batchsize = 16
    max_epoch = 5
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    patch_size = 16
    # One-tower settings
    model_type = "one-tower"
    # encoder_type = 'text'
    pooler_type = 'double' # 'double' or 'single'
    encoder = "google/electra-small-discriminator"
    random_init_encoder = False
    
@ex.named_config
def finetune_snli_onetower_electra_base():
    exp_name = "snli_onetower_electra_base"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    per_gpu_batchsize = 16
    max_epoch = 5
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    patch_size = 16
    # One-tower settings
    model_type = "one-tower"
    # encoder_type = 'text'
    pooler_type = 'double' # 'double' or 'single'
    encoder = "google/electra-base-discriminator"
    random_init_encoder = False
    

@ex.named_config
def finetune_snli_onetower_bert_base():
    exp_name = "snli_onetower_bert_base"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    per_gpu_batchsize = 16
    max_epoch = 5
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    patch_size = 16
    # One-tower settings
    model_type = "one-tower"
    # encoder_type = 'text'
    pooler_type = 'double' # 'double' or 'single'
    encoder = "google-bert/bert-base-uncased"
    random_init_encoder = False
    
@ex.named_config
def finetune_snli_onetower_swin():
    exp_name = "snli_onetower_swin"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    per_gpu_batchsize = 4
    max_epoch = 5
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    patch_size = 16
    # One-tower settings
    model_type = "one-tower"
    # encoder_type = 'text'
    pooler_type = 'double' # 'double' or 'single'
    encoder = "microsoft/swin-tiny-patch4-window7-224"
    random_init_encoder = False
    

@ex.named_config
def finetune_snli_onetower_vit_base():
    exp_name = "snli_onetower_vit_base"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    per_gpu_batchsize = 16
    max_epoch = 5
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    patch_size = 16
    # One-tower settings
    model_type = "one-tower"
    # encoder_type = 'text'
    pooler_type = 'double' # 'double' or 'single'
    encoder = "google/vit-base-patch16-224"
    random_init_encoder = False
    
    
@ex.named_config
def finetune_snli_twotower_vision_fr_text_fr():
    exp_name = "snli_vision_fr_text_fr"
    model_type = "two-tower"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    # Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True
    
@ex.named_config
def finetune_ref_twotower():
    exp_name = "ref_twotower"
    model_type = "two-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    batch_size = 5
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 40
    image_size = 224
    
@ex.named_config
def finetune_ref_twotower_deit_electra():
    exp_name = "ref_twotower_deit_electra"
    model_type = "two-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 15
    max_text_len = 40
    image_size = 224
    
    # Encoder Settings
    text_encoder = "google/electra-small-discriminator"
    image_encoder = "facebook/deit-tiny-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    


@ex.named_config
def finetune_ref_twotower_deit_electra_exp2():
    exp_name = "ref_twotower_deit_electra_exp2"
    model_type = "two-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 15
    max_text_len = 40
    image_size = 224
    
    # Encoder Settings
    text_encoder = "google/electra-small-discriminator"
    image_encoder = "facebook/deit-tiny-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 292
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

@ex.named_config
def finetune_twotower_electrasmall_swintiny():
    exp_name = "ref_twotower_electrasmall_swintiny"
    model_type = "two-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 15
    max_text_len = 40
    image_size = 224
    
    # Encoder Settings
    image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    text_encoder = "google/electra-small-discriminator"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 7
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

@ex.named_config
def finetune_ref_twotower_vitbase_electrabase():
    exp_name = "ref_twotower_vitbase_electrabase"
    model_type = "two-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 15
    max_text_len = 40
    image_size = 224
    
    # Encoder Settings
    text_encoder = "google/electra-base-discriminator"
    image_encoder = "google/vit-base-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1


@ex.named_config
def finetune_ref_twotower_deittiny_electratiny():
    exp_name = "ref_twotower_deittiny_electratiny"
    model_type = "two-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    lr_mult_cross_modal = 15
    max_text_len = 40
    image_size = 224
    
    # Encoder Settings
    text_encoder = "claytonfields/electra-tiny"
    image_encoder = "facebook/deit-tiny-patch16-224"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
@ex.named_config
def finetune_ref_twotower_dinos_tinybert():
    exp_name = "ref_twotower_dinos_tinybert"
    model_type = "two-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    batch_size = 5
    max_epoch = 5
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 40
    image_size = 224
    
    # Encoder Settings
    text_encoder = "huawei-noah/TinyBERT_General_4L_312D"
    image_encoder = "facebook/dino-vits16"
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    
@ex.named_config
def finetune_ref_onetower():
    exp_name = "ref_onetower"
    model_type = "one-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    max_text_len = 40
    image_size = 224
    
    
@ex.named_config
def finetune_ref_onetower_deitsmall():
    exp_name = "ref_onetower_deitsmall"
    model_type = "one-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    max_text_len = 40
    image_size = 224
    # Encoder Settings
    encoder = "facebook/deit-small-distilled-patch16-224"
    

@ex.named_config
def finetune_ref_onetower_manual_config_electrabase_exp1():
    exp_name = "ref_onetower_deitsmall"
    model_type = "one-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    max_text_len = 40
    image_size = 224
    # Encoder Settings
    model_type = "one-tower"
    encoder = "google/electra-base-discriminator"
    # Train encoder model from scratch
    random_init_encoder = True
    ## Manual Configuration
    encoder_manual_configuration = True
    hidden_size = 448
    num_heads = 4
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    embedding_size = 128
    
@ex.named_config
def finetune_ref_onetower_vit_base():
    exp_name = "ref_onetower_vit_base"
    model_type = "one-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    max_text_len = 40
    image_size = 224
    # One-tower settings
    model_type = "one-tower"
    pooler_type = 'double' # 'double' or 'single'
    encoder = "google/vit-base-patch16-224"
    random_init_encoder = False
    
@ex.named_config
def finetune_ref_onetower_bert_base():
    exp_name = "ref_onetower_bert_base"
    model_type = "one-tower"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Hardware Settings
    num_nodes = 1
    num_gpus = 2
    per_gpu_batchsize = 10
    # Training Setttings
    batch_size = 60
    max_epoch = 10
    max_steps = 10e6
    warmup_steps = 0.05
    draw_false_image = 0
    learning_rate = 7.5e-4
    lr_mult_head = 5
    max_text_len = 40
    image_size = 224
    # One-tower settings
    model_type = "one-tower"
    pooler_type = 'double' # 'double' or 'single'
    encoder = "google-bert/bert-base-uncased"
    random_init_encoder = False
    

@ex.named_config
def finetune_irtr_coco_clip_bert():
    exp_name = "irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    text_encoder = "bert-base-uncased"
    text_encoder_hidden_size = 768
    image_encoder = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_encoder_hidden_size = 768
    image_size = 384

@ex.named_config
def finetune_irtr_f30k_clip_bert():
    exp_name = "irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    text_encoder = "bert-base-uncased"
    text_encoder_hidden_size = 768
    image_encoder = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_encoder_hidden_size = 768
    image_size = 384

# ===================== Texto-Only Tasks ===================== #
@ex.named_config
def finetune_mrpc_twotower():
    exp_name = "mrpc_twotower"
    datasets = ["glue"]
    loss_names = _loss_names({"mrpc": 1})
    # Training Time
    max_epoch = 3
    max_steps = None
    
    image_size = 224
    patch_size = 16
    
    # Training Settings
    batch_size = 32
    per_gpu_batchsize = 32
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 128
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = True
    freeze_cross_modal_layers = True
    
@ex.named_config
def finetune_mrpc_twotower_deit_electra():
    exp_name = "mrpc_twotower_deit_electra"
    datasets = ["glue"]
    loss_names = _loss_names({"mrpc": 1})
    batch_size = 32
    per_gpu_batchsize = 32
    # Training Time
    max_epoch = 3
    max_steps = 10e6
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_size = 224
    patch_size = 16
    # Training Settings
    batch_size = 32
    warmup_steps = 0
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 128
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = True
    freeze_cross_modal_layers = True
    
# ===================== Two-Tower Vision Encoders ===================== #
@ex.named_config
def twotower_image_swin_tiny_patch4_window7_224():
    model_type = "two-tower"
    image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    patch_size = 4
    image_size = 224
    original_image_size = 224
    
@ex.named_config
def twotower_image_deit_tiny_patch16_224():
    model_type = "two-tower"
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_size = 224
    original_image_size = 224
    patch_size = 16

@ex.named_config
def twotower_encoder_dino16():
    model_type = "two-tower"
    image_encoder = "facebook/dino-vits16"
    image_size = 224
    patch_size = 16
    
# ===================== Two-Tower Text Encoders ===================== #
@ex.named_config
def twotower_text_roberta():
    model_type = "two-tower"
    text_encoder = "FacebookAI/roberta-base"
    vocab_size = 50265



@ex.named_config
def twotower_text_electra_small():
    model_type = "two-tower"
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522    
    
@ex.named_config
def twotower_text_electra_base():
    model_type = "two-tower"
    tokenizer = "google/electra-base-discriminator"
    vocab_size = 30522
    


# ===================== One-Tower Encoders ===================== #
@ex.named_config
def onetower_encoder_electra_small():
    model_type = "one-tower"
    encoder = "google/electra-small-discriminator"
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_size = 224
    patch_size = 16

@ex.named_config
def onetower_encoder_electra_base():
    model_type = "one-tower"
    encoder = "google/electra-base-discriminator"
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_size = 224
    patch_size = 16
    
@ex.named_config
def onetower_encoder_dino16():
    model_type = "one-tower"
    encoder = "facebook/dino-vits16"
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_size = 224
    patch_size = 16
    
# ===================== Random Augmentations ===================== #
@ex.named_config
def clip_randaug():
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]

@ex.named_config
def imagenet_randaug():
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    
# =========== Freeze of Un-Freeze Encoders for Training =========== #
@ex.named_config
def freeze_image():
    freeze_image_encoder = True

@ex.named_config
def freeze_text():
    freeze_text_encoder = True

@ex.named_config
def freeze_cross_modal():
    freeze_cross_modal_layers = True
    

@ex.named_config
def random_init_text():
    random_init_text_encoder = True

@ex.named_config
def random_init_vision():    
    random_init_vision_encoder = True
    
# ===================== Test Cases ===================== #
@ex.named_config
def test_case_mlm_itm_a():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=64 
    resume_from = ''
    # SNLI-VE
    exp_name = "test_mlm_itm_a"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    # Training Time
    max_epoch = 1
    max_steps = 5
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    # image_encoder = "microsoft/resnet-18"
    # image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    image_encoder_hidden_size = 192
    image_size = 224
    patch_size = 16
    # train_transform_keys = ["imagenet"]
    # val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 32
    warmup_steps = 0.1
    draw_false_image = 1
    draw_false_text = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    whole_word_masking = True
    mlm_prob = 0.15
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False


@ex.named_config
def test_case_mlm_itm_b():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=64 
    resume_from = ''
    # SNLI-VE
    exp_name = "test_case_mlm_itm"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    # Training Time
    max_epoch = 1
    max_steps = 100
    # Text Encoder
    # text_encoder = "google/electra-base-discriminator"
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    # text_encoder_hidden_size = 768
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    # image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    image_encoder = "facebook/deit-tiny-patch16-224"
    patch_size = 4
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # image_encoder_hidden_size = 768
    resolution_before = 224
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 1
    draw_false_text = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    whole_word_masking = True
    mlm_prob = 0.15
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True
    
@ex.named_config
def test_case_mlm_itm_c():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=64 
    load_path = ''
    # SNLI-VE
    exp_name = "test_case_mlm_itm"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10
    # Text Encoder
    text_encoder = "google/electra-base-discriminator"
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-small-patch16-224"
    # patch_size = 4
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # image_encoder_hidden_size = 768
    # resolution_before = 224
    # Training Settings
    batch_size = 16
    warmup_steps = 0.1
    draw_false_image = 1
    draw_false_text = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    whole_word_masking = True
    mlm_prob = 0.15
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True


# SNLI
@ex.named_config
def test_case_finetune_snli_a():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=32 
    # load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    load_path='/home/claytonfields/nlp/code/renaissance/result/test_mlm_itm_a_seed0_is224_ps16_bs32_pgbs64_te1_ts1/version_0/checkpoints/last.ckpt'
    # SNLI-VE
    exp_name = "test_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    original_image_size = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    
@ex.named_config
def test_one_tower_mlm_itm_manual_config_case_a():
    exp_name = "test_one_tower_mlm_itm_manual_config_case_a"
    seed = 0
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 10  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    # per_gpu_batchsize = 12
    
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=2
    resume_from = None
    
    # Transformer Setting
    model_type = "one-tower"
    encoder = "google/electra-small-discriminator"
    # Train encoder model from scratch
    random_init_encoder = True
    ## Manual Configuration
    encoder_manual_configuration = True
    hidden_size = 192
    num_heads = 4
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    embedding_size = 96
    
    # Image setting
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    
    # Text Setting
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0
    
    max_steps = 10
    

@ex.named_config
def test_case_eval_snli_a():
    # Settings
    test_only=True
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=64 
    # load_path = 'result/finetune_snli_mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/version_0/checkpoints/epoch\=4-step\=20684.ckpt deit_tiny_patch16_224'
    # load_path = '/home/claytonfields/nlp/code/meter/result/finetune_snli_seed0_from_epoch=43-step=898039/version_1/checkpoints/epoch=4-step=20684.ckpt'
    # load_path = '/home/claytonfields/nlp/code/meter/result/finetune_snli_mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/version_0/checkpoints/epoch=4-step=20684.ckpt'
    # SNLI-VE
    exp_name = "test_case_finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Training Time
    max_epoch = 1
    max_steps = 1000
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    resolution_before = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    
@ex.named_config
def test_case_finetune_snli_b():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=32 
    load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    # load_path='/home/claytonfields/nlp/code/renaissance/result/test_mlm_itm_a_seed0_is224_ps16_bs32_pgbs64_te1_ts1/version_0/checkpoints/last.ckpt'
    # SNLI-VE
    exp_name = "test_snli_b"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10
    # Text Encoder
    text_encoder = "google/electra-base-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 768
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    patch_size = 4
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_encoder_hidden_size = 768
    resolution_before = 224
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True

@ex.named_config
def test_case_eval_snli_b():
    # Settings
    test_only=True
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=64 
    # load_path = 'result/finetune_snli_mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/version_0/checkpoints/epoch\=4-step\=20684.ckpt deit_tiny_patch16_224'
    # SNLI-VE
    exp_name = "test_case_finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Training Time
    max_epoch = 1
    max_steps = 1000
    # Text Encoder
    text_encoder = "google/electra-base-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 768
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    patch_size = 4
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_encoder_hidden_size = 768
    resolution_before = 224
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True
    
@ex.named_config
def test_case_eval_nlvr2_a():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=16
    # load_path = 'result/finetune_snli_mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/version_0/checkpoints/epoch\=4-step\=20684.ckpt deit_tiny_patch16_224'
    # load_path = '/home/claytonfields/nlp/code/meter/result/finetune_snli_seed0_from_epoch=43-step=898039/version_1/checkpoints/epoch=4-step=20684.ckpt'
    # load_path = '/home/claytonfields/nlp/code/meter/result/finetune_snli_mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/version_0/checkpoints/epoch=4-step=20684.ckpt'
    # SNLI-VE
    exp_name = "test_nlvr2_deit_electra"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10
    model_type = "two-tower"
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_size = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 16
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    
@ex.named_config
def test_case_vqa_a():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=16
    # load_path = 'result/finetune_snli_mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/version_0/checkpoints/epoch\=4-step\=20684.ckpt deit_tiny_patch16_224'
    # load_path = '/home/claytonfields/nlp/code/meter/result/finetune_snli_seed0_from_epoch=43-step=898039/version_1/checkpoints/epoch=4-step=20684.ckpt'
    # load_path = '/home/claytonfields/nlp/code/meter/result/finetune_snli_mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/version_0/checkpoints/epoch=4-step=20684.ckpt'
    # SNLI-VE
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 64
    max_epoch = 1
    max_steps = 10
    # Training Time
    model_type = "two-tower"
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_size = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 16
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    
@ex.named_config
def test_case_finetune_mrpc_a():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=32 
    # load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    # SNLI-VE
    exp_name = "test_case_finetune"
    datasets = ["glue"]
    loss_names = _loss_names({"mrpc": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10e6
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 32
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    
@ex.named_config
def test_case_finetune_mrpc_onetower():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=32 
    # load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    # SNLI-VE
    exp_name = "test_case_mrpc_onetower"
    datasets = ["glue"]
    loss_names = _loss_names({"mrpc": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10e6
    # Text Encoder
    model_type = "one-tower"
    encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    
    # Training Settings
    batch_size = 32
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    
    
# Refcoco Reference Resolution
@ex.named_config
def test_case_finetune_refcoco_a():
    exp_name = "test_finetune_ref_case_a"
    datasets = ["refcoco"]
    # "itm": 0, "mlm": 0
    loss_names = _loss_names({"ref": 1})
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    batch_size = 5
    per_gpu_batchsize=5
    load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    resolution_before = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    draw_false_image = 0
    # Text Encoder Settings
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    draw_false_text = 0
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # Optimizer and Training Settings
    warmup_steps = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False

    # Downstream Setting
    get_recall_metric = False
    
@ex.named_config
def test_one_tower_mlm_itm_case_a():
    exp_name = "test_one_tower_mlm_itm_case_a"
    seed = 0
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 42  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    # per_gpu_batchsize = 12
    
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=2
    resume_from = None
    
    # Transformer Setting
    # encoder = "facebook/deit-tiny-patch16-224"
    # encoder = "FacebookAI/roberta-base"
    model_type = "one-tower"
    encoder = "google/electra-small-discriminator"
    hidden_size = 192
    num_heads = 4
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    
    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 224
    # max_image_len = -1
    patch_size = 16
    draw_false_image = 1
    image_only = False
    
    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0
    
    max_steps = 1

@ex.named_config
def test_case_one_tower_finetune_snli_a():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=2
    # batch_size=16
    # load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    # SNLI-VE
    exp_name = "test_case_finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10
    # Transformer Setting
    # encoder = "facebook/deit-tiny-patch16-224"
    # encoder = "FacebookAI/roberta-base"
    model_type = "one-tower"
    encoder = "google/electra-small-discriminator"
    # hidden_size = 192
    # num_heads = 4
    # num_layers = 12
    # mlp_ratio = 4
    # drop_rate = 0.1
    
    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    original_image_size = 224
    # max_image_len = -1
    patch_size = 16
    draw_false_image = 1
    image_only = False
    
    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 50
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

@ex.named_config
def test_case_one_tower_eval_snli_a():
    # Settings
    test_only=True
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=64 
    # load_path = 'result/finetune_snli_mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/version_0/checkpoints/epoch\=4-step\=20684.ckpt deit_tiny_patch16_224'
    load_path = '/home/claytonfields/nlp/code/meter/result/finetune_snli_seed0_from_epoch=43-step=898039/version_1/checkpoints/epoch=4-step=20684.ckpt'
    # load_path = '/home/claytonfields/nlp/code/meter/result/finetune_snli_mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/version_0/checkpoints/epoch=4-step=20684.ckpt'
    # SNLI-VE
    exp_name = "test_case_finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Training Time
    max_epoch = 1
    max_steps = 1000
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    resolution_before = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False


@ex.named_config
def test_case_finetune_mrpc_b():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=32 
    # load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    # SNLI-VE
    exp_name = "test_case_finetune_mrpc_a"
    datasets = ["glue"]
    # glue_task = 'mrpc'
    loss_names = _loss_names({"mrpc": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10e6
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 32
    warmup_steps = 0
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = False
    freeze_cross_modal_layers = True

@ex.named_config
def test_case_finetune_ref_a():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=2
    model_type = 'two-tower'
    # load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    # load_path = "/home/claytonfields/nlp/code/meter/result/test_one_tower_mlm_itm_case_a_seed0_from_/version_29/checkpoints/last.ckpt"
    # SNLI-VE
    exp_name = "test_case_finetune_ref"
    datasets = ["refcoco"]
    loss_names = _loss_names({"ref": 1})
    # Training Time
    max_epoch = 1
    max_steps = 20
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 32
    patch_size = 8
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 6
    warmup_steps = 0
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False


