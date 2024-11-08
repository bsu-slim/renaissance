import json
import pandas as pd
import pyarrow as pa
import os

import sys
import os.path as osp
import pickle

from tqdm import tqdm


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()
    
def process_ref(ref, path, Imgs, imgToAnns):
    img = Imgs[ref['image_id']]
    image_path = osp.join(path, img['file_name'])
    with open(image_path, "rb") as fp:
        binary = fp.read()

    # TODO: Process text
    sents = []
    for sent in ref['sentences']:
        sents.append(sent['sent'])
    
    anns= imgToAnns[ref['image_id']]
    ann_id = ref['ann_id']
    obj_ids = []
    bboxes = []
    for ann in anns:
        obj_ids.append(ann['id'])
        bboxes.append(ann['bbox'])
    label = obj_ids.index(ann_id)
        
    return [binary, sents, bboxes, label]


    


def write_refcoco(data_root, outfile_root, dataset = 'refcoco', splitBy = 'unc'):
# ROOT_DIR = osp.abspath(osp.dirname(__file__))
    DATA_DIR = osp.join(data_root, dataset)
    if dataset in ['refcoco', 'refcoco+', 'refcocog']:
        IMAGE_DIR = osp.join(data_root, 'images/mscoco/train2014')
    elif dataset == 'refclef':
        IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
    else:
        print ('No refer dataset is called [%s]' % dataset)
        sys.exit()
    
    # load refs from data/dataset/refs(dataset).json
    ref_file = osp.join(DATA_DIR, 'refs('+splitBy+').p')
    data = {}
    data['dataset'] = dataset
    data['refs'] = pickle.load(StrToBytes(open(ref_file, 'r')))
    
    # load annotations from data/dataset/instances.json
    instances_file = osp.join(DATA_DIR, 'instances.json')
    instances = json.load(open(instances_file, 'r'))
    data['images'] = instances['images']
    data['annotations'] = instances['annotations']
    data['categories'] = instances['categories']
    
    
    Anns, Imgs, imgToAnns = {}, {}, {}
    for ann in data['annotations']:
        Anns[ann['id']] = ann
        imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
    for img in data['images']:
        Imgs[img['id']] = img
        
    
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        if split == 'test':
            refs = [ref for ref in tqdm(data['refs']) if 'test' in ref['split']]
        elif split == 'train' or split == 'val':
            refs = [ref for ref in tqdm(data['refs']) if ref['split'] == split]
    
        item_list = [process_ref(ref, IMAGE_DIR, Imgs, imgToAnns) for ref in refs]
        df = pd.DataFrame(item_list, columns=['image', 'sentences', 'bboxes', 'labels'])
        table = pa.Table.from_pandas(df)
        
        os.makedirs(outfile_root, exist_ok=True)
        file_path = osp.join(outfile_root, f"refcoco_{splitBy}_{split}.arrow")
        with pa.OSFile(
            file_path, "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


if __name__ == '__main__':
    
    data_root = '/home/claytonfields/nlp/code/data/coco'  # contains refclef, refcoco, refcoco+, refcocog and images
    outfile_root = '/home/claytonfields/nlp/code/meter/data/arrow'
    dataset = 'refcoco' 
    splitBy = 'unc'

    write_refcoco(data_root, outfile_root)





