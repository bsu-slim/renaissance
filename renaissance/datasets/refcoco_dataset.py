from .base_dataset import BaseDataset
import io
from PIL import Image
import torch
import numpy as np
import pyarrow as pa
import random


class RefcocoDataset(BaseDataset):
    def __init__(self, *args, split="", max_bb = 20, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.max_bb = max_bb

        if split == "train":
            names = ['refcoco_unc_train']
        elif split == "val":
            # names = ["coco_caption_karpathy_val"]
            names = ['refcoco_unc_val']
        elif split == "test":
            names = ['refcoco_unc_test']

        super().__init__(*args, names=names, text_column_name="sentences", remove_duplicate=False, **kwargs)
        # self.filter_table()

    # Generalize padding funtion for use in different methods
    def __getitem__(self, index):
        max_bb = self.max_bb
        image_index, ref_index = self.index_mapper[index]
        try:
            label = self.table["labels"][image_index].as_py()
            image = np.array(self.get_raw_image(index))
            bboxes = self.table['bboxes'][image_index].as_py()
        except IndexError:
            print("Hello World")
            print("Index: ", index)
            print("Image Index: ", image_index)
            print("Ref Index: ", ref_index)
            return_dict = {
                # 'ann_id' : ann_id,
                'image' : [torch.zeros(max_bb,3,self.image_size,self.image_size)],#.to(self.device)],
                # 'obj_ids' : torch.tensor(obj_ids_total),#.to(self.device),
                'target' : 0,#.to(self.device),
                'text' : '',
                'text_ids' : torch.zeros(max_bb,self.max_text_len,dtype=torch.int8),#.to(self.device),
                'text_labels' : torch.zeros(max_bb,self.max_text_len,dtype=torch.int8),#.to(self.device),
                'text_masks' : torch.zeros(max_bb,self.max_text_len,dtype=torch.int8),#.to(self.device)
            }
            
            return return_dict
            
        if len(bboxes) > max_bb:
            truth = bboxes.pop(label)
            bboxes = random.choices(bboxes, k=max_bb-1)
            bboxes.append(truth)
            random.shuffle(bboxes)
            label = bboxes.index(truth)
        
        sub_images = []
        # print("Hello World")
        for bbox in bboxes:
            bbox = [int(b) for b in bbox]
            sub = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
            if sub is not None:
                try :
                    sub = self.processor(
                        sub, 
                        return_tensors='pt',
                        size={'height':self.image_size, 'width':self.image_size}
                    )['pixel_values'][0]
                    sub_images.append(sub.unsqueeze(0))
                except ValueError:
                    print("Index: ", index)
                    print("Image Index: ", image_index)
                    print("Ref Index: ", ref_index)
        num_sub_images = len(sub_images)
        num_pad = max_bb - num_sub_images 
        
        pad_image = torch.zeros(1,3,self.image_size,self.image_size)
        for _ in range(max_bb - num_sub_images):
            sub_images.append(pad_image)
        
        # text ids
        text = self.get_text(index)
        text_tokenized = text['text'][1]
        ids = torch.tensor(text_tokenized['input_ids'])
        repeat_ids = ids.repeat(num_sub_images,1)
        pad_ids =  torch.zeros(num_pad,self.max_text_len,dtype=torch.int8)
        text_ids = torch.cat((repeat_ids, pad_ids))#.to(torch.long)
        # text masks
        masks = torch.tensor(text_tokenized['attention_mask'])
        repeat_masks = masks.repeat(num_sub_images,1)
        pad_masks = torch.zeros(num_pad, self.max_text_len, dtype=torch.int8)
        text_masks = torch.cat((repeat_masks, pad_masks))#.to(torch.long)
        # text_labels
        labels = torch.full((self.max_text_len,),-100, dtype=torch.int8)
        repeat_labels = labels.repeat(num_sub_images, 1)
        pad_labels = torch.zeros(num_pad, self.max_text_len, dtype=torch.int8)
        text_labels = torch.cat((repeat_labels, pad_labels))#.to(torch.long)
        
        # target = self.table

        return_dict = {
            # 'ann_id' : ann_id,
            'image' : [torch.cat(sub_images)],#.to(self.device)],
            # 'obj_ids' : torch.tensor(obj_ids_total),#.to(self.device),
            'target' : label,#.to(self.device),
            'text' : text['text'][0],
            'text_ids' : text_ids,#.to(self.device),
            'text_labels' : text_labels,#.to(self.device),
            'text_masks' : text_masks,#.to(self.device)
        }
        
        return return_dict
    
    def filter_table(self):
        df = self.table.to_pandas()
        def check_len(item):
            return item.size <= self.max_bb
        sub = df[df['bboxes'].apply(check_len)]
        sub.reset_index(inplace=True, drop=True)
        self.table = pa.Table.from_pandas(sub)
    
    
    def collate(self, batch, mlm_collator=None):
        targets = []
        for b in batch:
            targets.append(b['target'])
        targets = torch.tensor(targets)
        return (batch, targets)