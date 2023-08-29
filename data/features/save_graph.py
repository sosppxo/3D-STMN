import json,os
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import torch
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('--split', type=str, help='split', required=True)
    parser.add_argument('--data_root',default='data/features', type=str, help='data root')
    parser.add_argument('--max_len', type=int, default=None, help='max length of the description')
    args = parser.parse_args()
    return args

def build_graph(nlp, description):
    depend = list(nlp.dependency_parse(description))  # 依存关系
    words = list(nlp.word_tokenize(description)) # 分词
    relations, heads, tails = map(list, zip(*depend))
    
    # print(heads)
    # print(tails)
    offset = 0
    cnt = 0
    for i in range(len(relations)):
        if relations[i] == 'ROOT':
            offset += cnt
            cnt = 0
            if offset > 0:
                heads[i] = root
            else:
                root = heads[i]
        else:
            if heads[i] >= cnt: cnt = heads[i]
            if tails[i] >= cnt: cnt = tails[i]
            heads[i] += offset
        tails[i] += offset
    return {'words':words, 'heads':np.array(heads), 'tails': np.array(tails), 'relations': relations}

def build_save_graph(nlp, data_root, split, max_len):
    scanrefer = json.load(open(os.path.join(data_root, 'ScanRefer', 'ScanRefer_filtered_'+split+'.json')))
    if os.path.exists(os.path.join(data_root, 'features', split, 'graph')) == False:
        os.makedirs(os.path.join(data_root, 'features', split, 'graph'))
    for data in tqdm(scanrefer):
        scene_id = data["scene_id"]
        object_id = int(data["object_id"])
        ann_id = int(data["ann_id"])
        token = data["token"]
        if max_len is not None:
            graph = build_graph(nlp, ' '.join(token[:max_len]))
        else:
            graph = build_graph(nlp, ' '.join(token))
        
        if max_len is not None:
            torch.save(graph, os.path.join(data_root, 'features', split, 'graph', scene_id+'_'+str(object_id).zfill(3)+'_'+str(ann_id).zfill(3)+'_max_len_'+str(max_len).zfill(3)+'.pth'))
        else:
            torch.save(graph, os.path.join(data_root, 'features', split, 'graph', scene_id+'_'+str(object_id).zfill(3)+'_'+str(ann_id).zfill(3)+'.pth'))

def main():
    args = get_args()
    # Starting Stanford CoreNLP Server
    nlp = StanfordCoreNLP('./stanford-corenlp-4.5.4', lang='en', memory='8g')

    split = args.split
    data_root = args.data_root
    max_len = args.max_len
    build_save_graph(nlp, data_root, split, max_len)
    nlp.close()

if __name__ == '__main__':
    main()