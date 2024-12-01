import os
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import argparse
import numpy
from nltk.corpus import wordnet as wn

arg = argparse.ArgumentParser()
arg.add_argument('--dataset', type=str, default='FB15k-237')

dataset = arg.parse_args().dataset
# read train.txt: head_id\trelation_id\ttail_id
train = pd.read_csv(os.path.join(dataset, 'train.txt'), sep='\t', header=None, names=['head_id', 'relation_id', 'tail_id'])
# read valid.txt: head_id\trelation\ttail_id
valid = pd.read_csv(os.path.join(dataset, 'valid.txt'), sep='\t', header=None, names=['head_id', 'relation_id', 'tail_id'])
# read test.txt: head_id\trelation\ttail_id
test = pd.read_csv(os.path.join(dataset, 'test.txt'), sep='\t', header=None, names=['head_id', 'relation_id', 'tail_id'])

# read relations from entities.dict: order\tid
if os.path.exists(os.path.join(dataset, 'entities.dict')):
    entities = pd.read_csv(os.path.join(dataset, 'entities.dict'), sep='\t', header=None, names=['order', 'id'])
else:
    entities = pd.DataFrame({'id': list(set(train['head_id'].unique()) | set(train['tail_id'].unique()))})  
# read entity2name: mid.json, like: "/m/06rf7": {
#     "name": "Schleswig-Holstein",
#     "description": "Schleswig-Holstein is Germany's northernmost state with Kiel as its capital and historical significance from duchies.  "
# }
if os.path.exists(os.path.join(dataset, 'mid.json')):
    with open(os.path.join(dataset, 'mid.json')) as f:
        mid = json.load(f)
else:
    mid = {}

# 太慢，直接用：#train = #head + #tail
# entities['#train'] = entities['id'].apply(lambda x: train[(train['head_id'] == x) | (train['tail_id'] == x)].shape[0])
print('calculating count')
entities['#valid'] = entities['id'].apply(lambda x: valid[(valid['head_id'] == x) | (valid['tail_id'] == x)].shape[0])
print('valid done')
entities['#test'] = entities['id'].apply(lambda x: test[(test['head_id'] == x) | (test['tail_id'] == x)].shape[0])
print('test done')

def find_synset_by_id(synset_id):
    possible_pos = ['n', 'v', 'a', 'r']  # Noun, Verb, Adjective, Adverb
    for pos in possible_pos:
        try:
            # Attempt to create a synset with the given POS
            synset = wn.synset_from_pos_and_offset(pos, int(synset_id))
            return {
                "pos": pos,                  # Found POS
                "name": synset.name(),       # Synset name
                "description": synset.definition(),  # Definition
            }
        except Exception:
            continue  # Ignore exceptions and try the next POS
    return {}  # No matching POS found

entity_dic = {}
for ix, entity in tqdm(entities.iterrows(), total=entities.shape[0]):
    entity = dict(entity)
    # print(entity)
    # remove order
    entity.pop('order', None) 
    id = entity['id']
    head_df = train[train['head_id'] == id]
    tail_df = train[train['tail_id'] == id]
    if id in mid:
        entity['name'] = mid[id]['name']
        entity['description'] = mid[id]['description']
    if 'WN18' in dataset:
        entity.update(find_synset_by_id(id))
    # group by head_id and iterate
    head_relations = defaultdict(int)
    tail_relations = defaultdict(int)
    for relation_id, group in head_df.groupby('relation_id'):
        head_relations[relation_id] = group.shape[0]
    for relation_id, group in tail_df.groupby('relation_id'):
        tail_relations[relation_id] = group.shape[0]

    entity_dic[entity['id']] = {
        **entity,
        '#head': head_df.shape[0],
        '#tail': tail_df.shape[0],
        'head_relations': head_relations,
        'tail_relations': tail_relations
    }
    # print(entity_dic[entity['id']])

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (numpy.bool_,)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


# Function to ensure JSON-compatible keys
def convert_key(key):
    if isinstance(key, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                        numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                        numpy.uint16, numpy.uint32, numpy.uint64)):
        return int(key)
    elif isinstance(key, (numpy.float_, numpy.float16, numpy.float32, numpy.float64)):
        return float(key)
    elif isinstance(key, (numpy.bool_,)):
        return bool(key)
    else:
        return str(key)  # Convert other types to string to ensure compatibility

# Convert numpy types in keys for entity_dic
entity_dic_converted = {convert_key(k): v for k, v in entity_dic.items()}

with open(os.path.join(dataset, 'entity.json'), 'w') as f:
    json.dump(entity_dic_converted, f, indent=4, cls=NumpyEncoder)
