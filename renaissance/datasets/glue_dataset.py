#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:37:04 2024

@author: claytonfields
"""

from .base_dataset import BaseDataset


class GlueDataset(BaseDataset):
    def __init__(self,*args,  task='', split='', max_text_length=128, **kwargs):

            
            # self.data_dict = load_dataset('glue', self.task, split=self.split).to_dict()
            
            
            if  split not in ["train", "val", "test"]:
                raise ValueError(f"{split} is not a recognized data split.")
            self.split = split
            # self.tokenizer = tokenizer
            self.task = task
            self.tasks = ["cola","mnli","mrpc","qnli","qqp","rte","sst2","stsb","wnli"]
            if self.task not in self.tasks:
                raise ValueError("The selected GLUE task is not supported.")
            super().__init__(*args,hugging_face=True, **kwargs)
            
            
            self.max_text_length = max_text_length
            
            # Consider dicionary apping
            if self.task in ["rte", "mrpc", "stsb", "wnli"]:
                self.sentence1 = self.data_dict['sentence1']
                self.sentence2 = self.data_dict['sentence2']
            elif self.task in ["cola", "sst2"]:
                self.sentence1 = self.data_dict['sentence']
                self.sentence2 = None
            elif self.task in ["qqp"]:
                self.sentence1 = self.data_dict["question1"]
                self.sentence2 = self.data_dict["question2'"]
            elif self.task in ["qnli"]:
                self.sentence1 = self.data_dict["question"]
                self.sentence2 = self.data_dict["sentence"]
            elif self.task in ["mnli"]:
                self.sentence1 = self.data_dict["premise"]
                self.sentence2 = self.data_dict["hypothesis"]
            self.label = self.data_dict['label']
            self.idx = self.data_dict["idx"]
        
            
    def __len__(self):
            return len(self.idx)

    def __getitem__(self, index):

        sent1 = self.sentence1[index]
        if self.sentence2:
            sent2 = self.sentence2[index]
        else:
            sent2 = None
        label = self.label[index]
        # idx = self.idx[index]

        ret = self.tokenizer(
            sent1, 
            sent2,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        ret = {k: v.squeeze() for k,v in ret.items()}
        ret['label'] = label
        return ret
    
    