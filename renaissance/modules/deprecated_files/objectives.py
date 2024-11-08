"""
Task implementations for vision language 
encoder models. Methods in objectives.py are used in 
renaissance/modules/renaissance_module.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pytorch_lightning import LightningModule

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from .dist_utils import all_gather

# Find a way to generalize logging fucntions
def log_metrics(pl_module: LightningModule, ret: Dict, metric_list: List, task: str, phase: str = None):
    
    batch_size = pl_module.hparams.config['per_gpu_batchsize']
    if phase == None:
        phase = "train" if pl_module.training else "val"
        
    for metric in metric_list:
        if metric == 'loss':
            value = getattr(pl_module, f"{phase}_{task}_{metric}")(ret[f"{task}_{metric}"])
        elif metric in  ['accuracy','score']:
            if task in ['snli', 'nlvr2']:
                value = getattr(pl_module, f'{phase}_{task}_{metric}')(
                    ret[f"{phase}_{task}_logits"], ret[f'{phase}_{task}_labels']
                )
            else:
                value = getattr(pl_module, f"{phase}_{task}_{metric}")(
                    ret[f"{task}_logits"], ret[f"{task}_labels"]
                )
        elif metric == 'f1':
            preds = ret[f'{task}_logits'].argmax(dim=-1)
            value = getattr(pl_module, f"{phase}_{task}_f1")(
                preds, ret["{task}_targets"]
            )
        pl_module.log(f"{task}/{phase}/{metric}", value, batch_size=batch_size)
        # pl_module.log(f"mlm/{phase}/accuracy", acc)
        
# ======================= VL-Understanding Tasks ======================= #

def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }
    
    task = 'mlm'
    metric_list = ['loss', 'accuracy']
    log_metrics(pl_module, ret, metric_list, task)
    

    # phase = "train" if pl_module.training else "val"
    # loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    # acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
    #     ret["mlm_logits"], ret["mlm_labels"]
    # )
    # pl_module.log(f"mlm/{phase}/loss", loss)
    # pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret

def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }
    
    task = 'itm'
    metric_list = ['loss', 'accuracy']
    log_metrics(pl_module, ret, metric_list, task)

    # phase = "train" if pl_module.training else "val"
    # loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    # acc = getattr(pl_module, f"{phase}_itm_accuracy")(
    #     ret["itm_logits"], ret["itm_labels"]
    # )
    # pl_module.log(f"itm/{phase}/loss", loss)
    # pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret

# Update for one-tower mehtods
def compute_ref(pl_module, batch):
    targets = batch[1]
    batch = batch[0]
    logit_list = []
    for i,b in enumerate(batch):

        infer_dict = pl_module.infer(b)
        logits = pl_module.ref_classifier(infer_dict['cls_feats'])
        logit_list.append(logits.reshape(1,-1))
            
    logit_tensor = torch.cat(logit_list)
    loss = F.cross_entropy(logit_tensor, targets)
                                                     
    ret = {
        "ref_loss" : loss,
        "ref_logits" : logit_tensor,
        "ref_targets" : targets
    }
    
    task = 'ref'
    metric_list = ['loss', 'accuracy']
    log_metrics(pl_module, ret, metric_list, task)
    # phase = "train" if pl_module.training else "val"
    # loss = getattr(pl_module, f"{phase}_ref_loss")(ret["ref_loss"])
    # acc = getattr(pl_module, f"{phase}_ref_accuracy")(
    #     ret["ref_logits"], ret["ref_targets"]
    # )
    # pl_module.log(f"ref/{phase}/loss", loss, batch_size=pl_module.hparams.config["batch_size"], sync_dist=True)
    # pl_module.log(f"ref/{phase}/accuracy", acc, batch_size=pl_module.hparams.config["batch_size"], sync_dist=True)
    
    return ret
  

def compute_snli(pl_module, batch):
    infer = pl_module.infer(
        batch, mask_text=False, mask_image=False, 
    )
    snli_logits = pl_module.snli_classifier(infer["cls_feats"])

    snli_labels = batch["labels"]
    snli_labels = torch.tensor(snli_labels).to(pl_module.device).long()
    snli_loss = F.cross_entropy(snli_logits, snli_labels.view(-1))
    
    

    ret = {
        "snli_loss": snli_loss,
        "train_snli_logits": snli_logits,
        "train_snli_labels": snli_labels,
    }
                
    task = 'snli'
    metric_list = ['loss', 'accuracy']
    # log_metrics(pl_module, ret, metric_list, task)

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        # task = 'snli'
        # metric_list = ['loss', 'accuracy']
        log_metrics(pl_module, ret, metric_list, task, phase=phase)

        # phase = "train" if pl_module.training else "val"
        # loss = getattr(pl_module, f"{phase}_snli_loss")(ret["snli_loss"])
        # acc = getattr(pl_module, f"{phase}_snli_accuracy")(
        #     ret["snli_logits"], ret["snli_labels"]
        # )
        # pl_module.log(f"snli/{phase}/loss", loss)
        # pl_module.log(f"snli/{phase}/accuracy", acc)
    else:
        # phase =
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            phase = 'dev'
            dev_snli_logits = ret["train_snli_logits"][dev_batches]
            dev_snli_labels = ret["train_snli_labels"][dev_batches]
            dev_snli_loss = F.cross_entropy(dev_snli_logits, dev_snli_labels)
            ret.update({
                f'{phase}_{task}_logits' : dev_snli_logits,
                f'{phase}_{task}_labels' : dev_snli_labels,
                f'{phase}_{task}_loss' : dev_snli_loss
            })
            log_metrics(pl_module, ret, metric_list, task, phase=phase)
            # task = 'dev_snli'
            # dev_loss = getattr(pl_module, f"dev_snli_loss")(
            #     F.cross_entropy(
            #         ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
            #     )
            # )
            # dev_acc = getattr(pl_module, f"dev_snli_accuracy")(
            #     ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
            # )
            # pl_module.log(f"snli/dev/loss", dev_loss)
            # pl_module.log(f"snli/dev/accuracy", dev_acc)
        if test_batches:
            phase = 'test'
            test_snli_logits = ret["train_snli_logits"][test_batches]
            test_snli_labels = ret["train_snli_labels"][test_batches]
            test_snli_loss = F.cross_entropy(test_snli_logits, test_snli_labels)
            ret.update({
                f'{phase}_{task}_logits' : test_snli_logits,
                f'{phase}_{task}_labels' : test_snli_labels,
                f'{phase}_{task}_loss' : test_snli_loss
            })
            log_metrics(pl_module, ret, metric_list, task, phase=phase)
            # test_loss = getattr(pl_module, f"dev_snli_loss")(
            #     F.cross_entropy(
            #         ret["snli_logits"][test_batches], 
            #         ret["snli_labels"][test_batches]
            #     )
            # )
            # test_acc = getattr(pl_module, f"test_snli_accuracy")(
            #     ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
            # )
            # pl_module.log(f"snli/test/loss", test_loss)
            # pl_module.log(f"snli/test/accuracy", test_acc)

    return ret

def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }
    
    # Commenting out to test, restor new code below when done!!!
    # task = 'vqa'
    # metric_list = ['loss', 'score']
    # log_metrics(pl_module, ret, metric_list, task)
    
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels.view(-1))

    ret = {
        "train_nlvr2_loss": nlvr2_loss,
        "train_nlvr2_logits": nlvr2_logits,
        "train_nlvr2_labels": nlvr2_labels,
    }
    
    task = 'nlvr2'
    metric_list = ['loss', 'accuracy']
    log_metrics(pl_module, ret, metric_list, task)

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        log_metrics(pl_module, ret, metric_list, task, phase=phase)
    #     loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
    #     acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
    #         ret["nlvr2_logits"], ret["nlvr2_labels"]
    #     )
    #     pl_module.log(f"nlvr2/{phase}/loss", loss)
    #     pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    # else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            phase = 'dev'
            dev_nlvr2_logits = ret["snli_logits"][dev_batches]
            dev_nlvr2_labels = ret["snli_labels"][dev_batches]
            dev_nlvr2_loss = F.cross_entropy(dev_nlvr2_logits, dev_nlvr2_labels)
            ret.update({
                f'{phase}_{task}_logits' : dev_nlvr2_logits,
                f'{phase}_{task}_labels' : dev_nlvr2_labels,
                f'{phase}_{task}_loss' : dev_nlvr2_loss
            })
            log_metrics(pl_module, ret, metric_list, task, phase=phase)
            # dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
            #     F.cross_entropy(
            #         ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            #     )
            # )
            # dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
            #     ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            # )
            # pl_module.log(f"nlvr2/dev/loss", dev_loss)
            # pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            phase = 'test'
            test_nlvr2_logits = ret["snli_logits"][test_batches]
            test_nlvr2_labels = ret["snli_labels"][test_batches]
            test_nlvr2_loss = F.cross_entropy(test_nlvr2_logits, test_nlvr2_labels)
            ret.update({
                f'{phase}_{task}_logits' : test_nlvr2_logits,
                f'{phase}_{task}_labels' : test_nlvr2_labels,
                f'{phase}_{task}_loss' : test_nlvr2_loss
            })
            log_metrics(pl_module, ret, metric_list, task, phase=phase)
            # test_loss = getattr(pl_module, f"test_nlvr2_loss")(
            #     F.cross_entropy(
            #         ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            #     )
            # )
            # test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
            #     ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            # )
            # pl_module.log(f"nlvr2/test/loss", test_loss)
            # pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret




# ======================= Text Only ======================= #

# add glue_task function to handle these cases
# consider text general text classifcation head


def compute_mrpc(pl_module, batch):
    mrpc_labels = batch.pop('label', None)
    hidden_state = pl_module.infer_text_only(batch)
    mrpc_logits = pl_module.mrpc_classifier(hidden_state)
    mrpc_loss = F.cross_entropy(mrpc_logits, mrpc_labels)
    
    ret = {
        'mrpc_logits' : mrpc_logits,
        'mrpc_targets' : mrpc_labels,
        'mrpc_loss' : mrpc_loss
    }
    
    task = 'mrpc'
    metric_list = ['loss', 'accuracy','f1']
    log_metrics(pl_module, ret, metric_list, task)
    
    return ret

def compute_rte(pl_module, batch):
    rte_labels = batch.pop('label', None)
    cls_feat = pl_module.infer_text_only(batch)
    rte_logits = pl_module.rte_classifier(cls_feat)
    rte_loss = F.cross_entropy(rte_logits, rte_labels)
    
    ret = {
        'rte_logits' : rte_logits,
        'rte_targets' : rte_labels,
        'rte_loss' : rte_loss
    }
    
    task = 'rte'
    metric_list = ['loss', 'accuracy']
    log_metrics(pl_module, ret, metric_list, task)
    
    return ret

def compute_wnli(pl_module, batch):
    wnli_labels = batch.pop('label', None)
    cls_feat = pl_module.infer_text_only(batch)
    wnli_logits = pl_module.wnli_classifier(cls_feat)
    wnli_loss = F.cross_entropy(wnli_logits, wnli_labels)
    
    ret = {
        'wnli_logits' : wnli_logits,
        'wnli_targets' : wnli_labels,
        'wnli_loss' : wnli_loss
    }
    
    task = 'wnli'
    metric_list = ['loss', 'accuracy']
    log_metrics(pl_module, ret, metric_list, task)
    
    return ret

def compute_sst2(pl_module, batch):
    sst2_labels = batch.pop('label', None)
    cls_feat = pl_module.infer_text_only(batch)
    sst2_logits = pl_module.sst2_classifier(cls_feat)
    sst2_loss = F.cross_entropy(sst2_logits, sst2_labels)
    
    ret = {
        'sst2_logits' : sst2_logits,
        'sst2_targets' : sst2_labels,
        'sst2_loss' : sst2_loss
    }
    
    task = 'sst2'
    metric_list = ['loss', 'accuracy']
    log_metrics(pl_module, ret, metric_list, task)
    
    return ret

def compute_qqp(pl_module, batch):
    qqp_labels = batch.pop('label', None)
    cls_feat = pl_module.infer_text_only(batch)
    qqp_logits = pl_module.qqp_classifier(cls_feat)
    qqp_loss = F.cross_entropy(qqp_logits, qqp_labels)
    
    ret = {
        'qqp_logits' : qqp_logits,
        'qqp_targets' : qqp_labels,
        'qqp_loss' : qqp_loss
    }
    
    task = 'qqp'
    metric_list = ['loss', 'accuracy','f1']
    log_metrics(pl_module, ret, metric_list, task)
    
    return ret

def compute_qnli(pl_module, batch):
    qnli_labels = batch.pop('label', None)
    cls_feat = pl_module.infer_text_only(batch)
    qnli_logits = pl_module.qnli_classifier(cls_feat)
    qnli_loss = F.cross_entropy(qnli_logits, qnli_labels)
    
    ret = {
        'qnli_logits' : qnli_logits,
        'qnli_targets' : qnli_labels,
        'qnli_loss' : qnli_loss
    }
    
    task = 'qnli'
    metric_list = ['loss', 'accuracy']
    log_metrics(pl_module, ret, metric_list, task)
    
    return ret

def compute_mnli(pl_module, batch):
    mnli_labels = batch.pop('label', None)
    cls_feat = pl_module.infer_text_only(batch)
    mnli_logits = pl_module.mnli_classifier(cls_feat)
    mnli_loss = F.cross_entropy(mnli_logits, mnli_labels)
    
    ret = {
        'mnli_logits' : mnli_logits,
        'mnli_targets' : mnli_labels,
        'mnli_loss' : mnli_loss
    }
    
    task = 'mnli'
    metric_list = ['loss', 'accuracy']
    log_metrics(pl_module, ret, metric_list, task)
    
    return ret

def compute_cola(pl_module, batch):
    cola_labels = batch.pop('label', None)
    cls_feat = pl_module.infer_text_only(batch)
    cola_logits = pl_module.cola_classifier(cls_feat)
    cola_loss = F.cross_entropy(cola_logits, cola_labels)
    
    ret = {
        'cola_logits' : cola_logits,
        'cola_targets' : cola_labels,
        'cola_loss' : cola_loss
    }
    
    task = 'cola'
    metric_list = ['loss', 'accuracy']
    log_metrics(pl_module, ret, metric_list, task)
    
    return ret

# Create image classification task
def compute_cifar10(pl_module, batch):
    image = batch['image']
    hidden_state = pl_module.image_encoder(image).last_hidden_state#.squeeze()
    cls_feat = pl_module.image_classification_pooler(hidden_state)
    cifar10_logits = pl_module.cifar10_classifier(cls_feat)
    cifar10_labels = batch['label']
    cifar10_loss = F.cross_entropy(cifar10_logits, cifar10_labels)
    
    ret = {
        'cifar10_logits' : cifar10_logits,
        'cifar10_targets' : cifar10_labels,
        'cifar10_loss' : cifar10_loss
    }
    
    task = 'cifar10'
    metric_list = ['loss', 'accuracy']
    log_metrics(pl_module, ret, metric_list, task)
    
    return ret


# ======================= Recall ======================= #

def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.batchmodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    #TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        img=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)
    
# ======================= Utility Functions ======================= #

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    try:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
            if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
        )
    except:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["gqa_test"].id2answer
            if "gqa_test" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["gqa"].id2answer
        )
        vqa_logits = output["vqa_logits"]
        vqa_preds = vqa_logits.argmax(dim=-1)
        vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
        questions = batch["text"]
        qids = batch["qid"]
        return {"qids": qids, "preds": vqa_preds, "gqa": True}
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds, "gqa": False}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    distributed = torch.distributed.is_initialized()
    if distributed:
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    qids, preds = list(), list()
    gqa = False
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
        gqa = out['gqa']

    rets = list()
    for qid, pred in zip(qids, preds):
        if gqa:
            rets.append({"questionId": qid, "prediction": pred})
        else:
            rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    if distributed:
        torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    if distributed:    
        torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
