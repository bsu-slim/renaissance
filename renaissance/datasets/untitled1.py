#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:12:41 2024

@author: claytonfields
"""

import json
import pandas as pd
import pyarrow as pa
import os

from tqdm import tqdm
from collections import defaultdict


data_dir = '/home/claytonfields/nlp/code/meter/data/arrow'
transform_keys = ['imagenet']
image_size = 224
# names = []
names = ["snli_dev", "snli_test"]
# text_column_name = ""
text_column_name="sentences"
remove_duplicate=True
max_text_len=40
draw_false_image=0
draw_false_text=0
image_only=False
tokenizer=None

"""
data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
transform_keys : keys for generating augmented views of images
text_column_name : pyarrow table column name that has list of strings as elements
"""
assert len(transform_keys) >= 1

# transforms = keys_to_transforms(transform_keys, size=image_size)
# clip_transform = False
# for transform_key in transform_keys:
#     if 'clip' in transform_key:
#         clip_transform = True
#         break
text_column_name = text_column_name
names = names
max_text_len = max_text_len
draw_false_image = draw_false_image
draw_false_text = draw_false_text
image_only = image_only
data_dir = data_dir

if len(names) != 0:
    tables = [
        pa.ipc.RecordBatchFileReader(
            pa.memory_map(f"{data_dir}/{name}.arrow", "r")
        ).read_all()
        for name in names
        if os.path.isfile(f"{data_dir}/{name}.arrow")
    ]

    table_names = list()
    for i, name in enumerate(names):
        table_names += [name] * len(tables[i])

    table = pa.concat_tables(tables, promote=True)
    if text_column_name != "":
        text_column_name = text_column_name
        all_texts = table[text_column_name].to_pandas().tolist()
        if type(all_texts[0][0]) == str:
            all_texts = (
                [list(set(texts)) for texts in all_texts]
                if remove_duplicate
                else all_texts
            )
        else: #snli
            all_texts = (
                [[t[1].strip() for t in texts] for texts in all_texts]
            )
    else:
        all_texts = list()
else:
    all_texts = list()

index_mapper = dict()

if text_column_name != "" and not image_only:
    j = 0
    for i, texts in enumerate(all_texts):
        for _j in range(len(texts)):
            index_mapper[j] = (i, _j)
            j += 1
else:
    for i in range(len(table)):
        index_mapper[i] = (i, None)