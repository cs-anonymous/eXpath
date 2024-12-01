import json
import os
from collections import defaultdict
import pandas as pd

dataset2config = {
    'FB15k': {
        'alpha': 0.01,
        'setting': 'score'
    },
    'FB15k-237': {
        'alpha': 0.01,
        'setting': 'score'
    },
    'WN18': {
        'alpha': 0.01,
        'setting': 'score_h'
    },
    'WN18RR': {
        'alpha': 0.01,
        'setting': 'score_h'
    }
}

files = []
model_dataset_list = []
for model in ['complex', 'conve', 'transe']:
    for dataset in ['FB15k', 'WN18', 'FB15k-237', 'WN18RR']: 
        config = dataset2config[dataset]
        alpha = config['alpha']
        files.append(f'out(alpha={alpha})/{model}_{dataset}/output_end_to_end_{config["setting"]}4.json')
        model_dataset_list.append(f'{model}_{dataset}')

rx = pd.DataFrame(0, columns=model_dataset_list, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'eXpath', 'data_poisoning+eXpath', 'criage+eXpath', 'k1+eXpath', 'kelpie+eXpath'])
rx_h = pd.DataFrame(0, columns=model_dataset_list, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'eXpath', 'data_poisoning+eXpath', 'criage+eXpath', 'k1+eXpath', 'kelpie+eXpath'])
rx_t = pd.DataFrame(0, columns=model_dataset_list, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'eXpath', 'data_poisoning+eXpath', 'criage+eXpath', 'k1+eXpath', 'kelpie+eXpath'])
fx = pd.DataFrame(0, columns=model_dataset_list, index=['data_poisoning', 'criage', 'k1', 'kelpie'])
gx = pd.DataFrame(0, columns=model_dataset_list, index=['data_poisoning', 'criage', 'k1', 'kelpie'])

def process(file):
    folder = os.path.dirname(file)
    model_dataset = folder.split('/')[-1]
    with open(file, 'r') as f:
        data = json.load(f)
        print('processing', file, len(data))

    # return
    base_prediction2data = {','.join(t['prediction']) :t for t in data if 'prediction' in t}
    base_prediction2data_subset = {k: v for k, v in base_prediction2data.items() if v['dMRR'] > 0.1}
    
    process_record(base_prediction2data, 'eXpath', model_dataset)
    process_record(base_prediction2data_subset, 'eXpath(subset)', model_dataset)

    # print(base_prediction2data)
    
    for setting in ['data_poisoning', 'criage', 'k1', 'kelpie']:
        file = f'output_end_to_end_{setting}4.json'
        if not os.path.exists(f'{folder}/{file}'):
            print(f'{folder}/{file} not exists')
            continue
        
        print('processing', folder, setting)
        with open(f'{folder}/{file}', 'r') as f:
            data = json.load(f)
        prediction2data = {','.join(t['prediction']) :t for t in data if 'prediction' in t}
        prediction2data_subset = {k: v for k, v in prediction2data.items() if k in base_prediction2data_subset}

        vxy = 0
        sxy = 0
        gxy_m = 0
        gxy_n = 0
        for k in prediction2data:
            if base_prediction2data[k]['dMRR'] > 0 or prediction2data[k]['dMRR'] > 0:
                vxy += 1
                if base_prediction2data[k]['dMRR'] > prediction2data[k]['dMRR']:
                    sxy += 1
                    gxy_m += base_prediction2data[k]['dMRR'] - prediction2data[k]['dMRR']
                    gxy_n += prediction2data[k]['dMRR']
        fx.loc[setting, model_dataset] = round(sxy / vxy, 3)
        gx.loc[setting, model_dataset] = round(gxy_m / gxy_n, 3)

        process_record(prediction2data, setting, model_dataset)
        process_record(prediction2data_subset, f'{setting}(subset)', model_dataset)

        prediction2data = {k: greater_metric(v, base_prediction2data[k]) for k, v in prediction2data.items()}
        prediction2data_subset = {k: v for k, v in prediction2data.items() if k in base_prediction2data_subset}

        process_record(prediction2data, setting+'+eXpath', model_dataset)
        process_record(prediction2data_subset, setting+'+eXpath(subset)', model_dataset)

        


def greater_metric(v1, v2):
    if v1['dMRR'] > v2['dMRR']:
        return v1
    return v2

def process_record(prediction2data, setting, model_dataset):
    dMRR = 0
    dMRR_h = 0
    dMRR_t = 0
    MRR = 0
    MRR_h = 0
    MRR_t = 0
    for k, v in prediction2data.items():
        if 'prediction' not in v:
            continue
        if 'original' not in v:
            continue
        dMRR += v['dMRR']
        MRR += v['original']['MRR']
        dMRR_h += v['original']['MRR_head'] - v['new']['MRR_head']
        MRR_h += v['original']['MRR_head']
        dMRR_t += v['original']['MRR_tail'] - v['new']['MRR_tail']
        MRR_t += v['original']['MRR_tail']

    rx.loc[setting, model_dataset] = round(dMRR / MRR, 3)
    rx_h.loc[setting, model_dataset] = round(dMRR_h / MRR_h, 3)
    rx_t.loc[setting, model_dataset] = round(dMRR_t / MRR_t, 3)
    

for file in files:
    process(file)

os.makedirs('metric', exist_ok=True)
rx.to_csv('metric/rx.csv')
rx_h.to_csv('metric/rx_h.csv')
rx_t.to_csv('metric/rx_t.csv')
fx.to_csv('metric/fx.csv')
gx.to_csv('metric/gx.csv')