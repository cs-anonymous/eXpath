import os
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import argparse

arg = argparse.ArgumentParser()
arg.add_argument('--dataset', type=str, default='FB15k-237')

dataset = arg.parse_args().dataset

# read relations from relations.dict: order\tid
relations = pd.read_csv(os.path.join(dataset, 'relations.dict'), sep='\t', header=None, names=['order', 'id'])
relations['relation'] = relations['id'].apply(lambda x: x.split('/')[-1])

# read train.txt: head_id\trelation_id\ttail_id
train = pd.read_csv(os.path.join(dataset, 'train.txt'), sep='\t', header=None, names=['head_id', 'relation_id', 'tail_id'])

# read valid.txt: head_id\trelation\ttail_id
valid = pd.read_csv(os.path.join(dataset, 'valid.txt'), sep='\t', header=None, names=['head_id', 'relation_id', 'tail_id'])

# read test.txt: head_id\trelation\ttail_id
test = pd.read_csv(os.path.join(dataset, 'test.txt'), sep='\t', header=None, names=['head_id', 'relation_id', 'tail_id'])

relations['#train'] = relations['id'].apply(lambda x: train[train['relation_id'] == x].shape[0])
relations['#valid'] = relations['id'].apply(lambda x: valid[valid['relation_id'] == x].shape[0])
relations['#test'] = relations['id'].apply(lambda x: test[test['relation_id'] == x].shape[0])
relations['#source'] = relations['id'].apply(lambda x: train[train['relation_id'] == x]['head_id'].nunique())
relations['#target'] = relations['id'].apply(lambda x: train[train['relation_id'] == x]['tail_id'].nunique())


# read relations from entities.dict: order\tid
# entities = pd.read_csv(os.path.join(dataset, 'entities.dict'), sep='\t', header=None, names=['order', 'id'])
# read entity2name: mid.json, like: "/m/06rf7": {
#     "name": "Schleswig-Holstein",
#     "description": "Schleswig-Holstein is Germany's northernmost state with Kiel as its capital and historical significance from duchies.  "
# }
# if os.path.exists(os.path.join(dataset, 'mid.json')):
#     with open(os.path.join(dataset, 'mid.json')) as f:
#         mid = json.load(f)

#     entities['entity'] = entities['id'].apply(lambda x: mid[x]['name'])
#     entities['description'] = entities['id'].apply(lambda x: mid[x]['description'])
# else:
#     entities['entity'] = entities['id']
#     entities['description'] = ''

# entities['#train'] = entities['id'].apply(lambda x: train[(train['head_id'] == x) | (train['tail_id'] == x)].shape[0])
# entities['#valid'] = entities['id'].apply(lambda x: valid[(valid['head_id'] == x) | (valid['tail_id'] == x)].shape[0])
# entities['#test'] = entities['id'].apply(lambda x: test[(test['head_id'] == x) | (test['tail_id'] == x)].shape[0])
# entities['#source'] = entities['id'].apply(lambda x: train[train['head_id'] == x].shape[0])
# entities['#target'] = entities['id'].apply(lambda x: train[train['tail_id'] == x].shape[0])

with open(os.path.join(dataset, 'relation_name.json')) as f:
    relation2name = json.load(f)
    
relation_dic = {}
for relation_id in tqdm(relations['id']):
    relation_df = train[train['relation_id'] == relation_id]
    # group by head_id and iterate
    count1toN = defaultdict(int)
    countNto1 = defaultdict(int)
    for head_id, group in relation_df.groupby('head_id'):
        count1toN[group.shape[0]] += 1
    for tail_id, group in relation_df.groupby('tail_id'):
        countNto1[group.shape[0]] += 1

    relation_dic[relation_id] = {
        'name': relation2name.get(relation_id, ''),
        'count1toN': count1toN,
        'countNto1': countNto1,
        **relations[relations['id'] == relation_id].to_dict(orient='records')[0]
    }

with open(os.path.join(dataset, 'relation.json'), 'w') as f:
    json.dump(relation_dic, f, indent=4)