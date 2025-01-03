import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np


# base_dir = 'out(alpha=0.05)'
# base_dir = 'outV7-T1'
# base_dir = 'out'
# base_dir = 'out'

# 'outV8-T1', 'outV8-T2', 'outV8-T3', 'outV8-T4', 
base_dirs = ['out']
folders = []
for model in ['complex', 'conve', 'transe']:
    for dataset in ['FB15k', 'FB15k-237', 'WN18', 'WN18RR']:    # , 'YAGO3-10'
        folder = f'{model}_{dataset}'
        folders.append(folder)
        
rx = pd.DataFrame(0, columns=folders, index=['criage', 'DP', 'k1', 'kelpie'])
rx_h = pd.DataFrame(0, columns=folders, index=['criage', 'DP', 'k1', 'kelpie'])
rx_t = pd.DataFrame(0, columns=folders, index=['criage', 'DP', 'k1', 'kelpie'])
fx = pd.DataFrame(columns=folders)
gx = pd.DataFrame(columns=folders)
DHit1 = pd.DataFrame(columns=folders, index=['criage', 'DP', 'k1', 'kelpie'])
DHit1_h = pd.DataFrame(columns=folders, index=['criage', 'DP', 'k1', 'kelpie'])
DHit1_t = pd.DataFrame(columns=folders, index=['criage', 'DP', 'k1', 'kelpie'])
best_count = pd.DataFrame(0, columns=folders, index=['eXpath()1', 'data_poisoning', 'k1', 'KGEAttack'])

def setting2path(folder, setting):
    suffix = setting
    if 'WN18' in folder:
        if setting in ['eXpath(011)', 'eXpath(101)', 'eXpath(110)', 'eXpath(100)']:
            suffix = setting.replace('(', '(h')
        elif setting == 'eXpath':
            suffix = 'eXpath(h)'
    else:
        if setting == 'eXpath':
            suffix = 'eXpath()'
    
    file = f'output_end_to_end_{suffix}4.json'
    if setting == 'eXpath1':
        if 'WN18' in folder:
            file = f'output_end_to_end_eXpath(h)1.json'
        else:
            file = f'output_end_to_end_eXpath()1.json'
    return f'{folder}/{file}'


def getAverage(dics):
    ret = defaultdict(float)
    for dic in dics:
        for k, v in dic.items():
            ret[k] += v
    for k, v in ret.items():
        ret[k] = v / len(dics)
    return ret

def loadAverageData(folder, setting):
    # path = setting2path(folder, setting)
    if 'eXpath' in setting:
        if '(' not in setting:
            setting = setting.replace('eXpath', 'eXpath()')
            if 'WN18' in folder:
                setting = setting.replace('()', '(h)')
        if ('(0' in setting or '(1' in setting)  \
            and 'WN18' in folder:
            setting = setting.replace('(', '(h')
        path = f'{folder}/output_end_to_end_{setting}.json'
    else:
        path = f'{folder}/output_end_to_end_{setting}4.json'
    datas = []
    for base_dir in base_dirs:
        if os.path.exists(f'{base_dir}/{path}'):
            with open(f'{base_dir}/{path}', 'r') as f:
                data = json.load(f)
            datas.append({','.join(t['prediction']) :t for t in data if 'prediction' in t})
    
    print('load', path, len(datas))
    if len(datas) == 0:
        return None
    prediction2data = {}
    for k in datas[0]:
        prediction2data[k] = datas[0][k]
        prediction2data[k]['original'] = getAverage([d.get(k)['original'] for d in datas])
        prediction2data[k]['new'] = getAverage([d.get(k)['new'] for d in datas])
        prediction2data[k]['dMRR'] = prediction2data[k]['original']['MRR'] - prediction2data[k]['new']['MRR']
    
    return prediction2data


def process(folder):
    for setting in ['criage', 'DP', 'k1', 'KGEAttack', 'kelpie', \
                    'eXpath(011)1', 'eXpath(101)1', 'eXpath(110)1', 'eXpath(100)1', 'eXpath(001)1', 'eXpath(000)1', \
                    'eXpath(011)4', 'eXpath(101)4', 'eXpath(110)4', 'eXpath(100)4', 'eXpath(001)4', 'eXpath(000)4', \
                    'eXpath()1', 'eXpath(h)1', 'eXpath(t)1', 'eXpath1', \
                    'eXpath()2', 'eXpath(h)2', 'eXpath(t)2', 'eXpath2', \
                    'eXpath()4', 'eXpath(h)4', 'eXpath(t)4', 'eXpath4', \
                    'eXpath()8', 'eXpath(h)8', 'eXpath(t)8', 'eXpath8', \
                    'criage+eXpath1', 'DP+eXpath1', 'k1+eXpath1', 'KGEAttack+eXpath1', \
                    'kelpie+eXpath4']:
        print('processing', folder, setting)
        _setting = setting.replace('DP', 'data_poisoning')
        if '+' in setting:
            # file1 = setting2path(folder, _setting.split('+')[0])
            # file2 = setting2path(folder, _setting.split('+')[1])
            # if not os.path.exists(file1) or not os.path.exists(file2):
            #     print(file1, file2, 'not exists')
            #     continue
            # with open(file1, 'r') as f:
            #     data1 = json.load(f)
            # with open(file2, 'r') as f:
            #     data2 = json.load(f)
            # prediction2data1 = {','.join(t['prediction']) :t for t in data1 if 'prediction' in t}
            # prediction2data2 = {','.join(t['prediction']) :t for t in data2 if 'prediction' in t}
            
            prediction2data1 = loadAverageData(folder, _setting.split('+')[0])
            prediction2data2 = loadAverageData(folder, _setting.split('+')[1])
            if prediction2data1 is None or prediction2data2 is None:
                continue
            prediction2data = {}
            for k in set(prediction2data1.keys()) | set(prediction2data2.keys()):
                if prediction2data2.get(k, {'dMRR': 0})['dMRR'] > prediction2data1.get(k, {'dMRR': 0})['dMRR']:
                    prediction2data[k] = prediction2data2[k]
                else:
                    prediction2data[k] = prediction2data1[k]
        else:
            # file = setting2path(folder, _setting)
            # if not os.path.exists(file):
            #     print(file, 'not exists')
            #     continue
            # with open(file, 'r') as f:
            #     data = json.load(f)
            # prediction2data = {','.join(t['prediction']) :t for t in data if 'prediction' in t}
            prediction2data = loadAverageData(folder, _setting)
            if prediction2data is None:
                continue

        dMRR = 0
        dMRR_h = 0
        dMRR_t = 0
        MRR = 0
        MRR_h = 0
        MRR_t = 0
        Hit1_original_h = 0
        DHit1_new_h = 0
        Hit1_original_t = 0
        DHit1_new_t = 0
        for k, v in prediction2data.items():
            if 'prediction' not in v:
                continue
            if 'original' not in v:
                continue
            dMRR += v['dMRR'] # max(v['dMRR'], 0)
            MRR += v['original']['MRR']
            dMRR_h += v['original']['MRR_head'] - v['new']['MRR_head']
            MRR_h += v['original']['MRR_head']
            dMRR_t += v['original']['MRR_tail'] - v['new']['MRR_tail']
            MRR_t += v['original']['MRR_tail']
            if v['original']['rank_head'] == 1:
                Hit1_original_h += 1
                if v['new']['rank_head'] > v['original']['rank_head'] + 0.5:
                    DHit1_new_h += 1
            if v['original']['rank_tail'] == 1:
                Hit1_original_t += 1
                if v['new']['rank_tail'] > v['original']['rank_tail'] + 0.5:
                    DHit1_new_t += 1

        DHit1_h.loc[setting, folder] = round(DHit1_new_h / Hit1_original_h, 3)
        DHit1_t.loc[setting, folder] = round(DHit1_new_t / Hit1_original_t, 3)
        DHit1.loc[setting, folder] = round((DHit1_new_h+DHit1_new_t) / (Hit1_original_h+Hit1_original_t), 3)
        rx.loc[setting, folder] = round(dMRR / MRR, 3) 
        rx_h.loc[setting, folder] = round(dMRR_h / MRR_h, 3) 
        rx_t.loc[setting, folder] = round(dMRR_t / MRR_t, 3) 

    for pair in [('eXpath()1', 'criage'), ('eXpath()1', 'data_poisoning'), ('eXpath()1', 'k1'), \
                 ('eXpath()1', 'KGEAttack'), ('eXpath(h)4', 'kelpie')]:
        setting1, setting2 = pair
        # file1 = setting2path(folder, setting1)
        # file2 = setting2path(folder, setting2)
        # if not os.path.exists(file1) or not os.path.exists(file2):
        #     print(file1, file2, 'not exists')
        #     continue
        # with open(file1, 'r') as f:
        #     data1 = json.load(f)
        # with open(file2, 'r') as f:
        #     data2 = json.load(f)
        # prediction2data1 = {','.join(t['prediction']) :t for t in data1 if 'prediction' in t}
        # prediction2data2 = {','.join(t['prediction']) :t for t in data2 if 'prediction' in t}
        prediction2data1 = loadAverageData(folder, setting1)
        prediction2data2 = loadAverageData(folder, setting2)
        if prediction2data1 is None or prediction2data2 is None:
            continue

        vxy = 0
        sxy = 0
        gxy_m = 0
        gxy_n = 0
        for k in prediction2data2:
            if prediction2data1[k]['dMRR'] > 0 or prediction2data2[k]['dMRR'] > 0:
                vxy += 1
                original = prediction2data1[k]['original']['MRR']
                if (prediction2data1[k]['dMRR'] - prediction2data2[k]['dMRR']) / original > 0.01:
                # if prediction2data1[k]['dMRR'] > prediction2data2[k]['dMRR']:
                    sxy += 1
                    gxy_m += prediction2data1[k]['dMRR'] - prediction2data2[k]['dMRR']
                    gxy_n += prediction2data2[k]['dMRR']
        # fx.loc[setting2, folder] = round(sxy / vxy, 3)
        # gx.loc[setting2, folder] = round(gxy_m / gxy_n, 3)
        fx.loc[setting2, folder] = round(sxy / vxy * 1000) / 10
        # gx.loc[setting2, folder] = str(round(gxy_m / gxy_n * 1000) / 10) + '%'


    # 计算最优的setting的percentage
    # best_prediction2data = loadAverageData(folder, 'eXpath()1')
    # for setting in ['data_poisoning', 'k1', 'KGEAttack']:
    #     prediction2data = loadAverageData(folder, setting)
    #     if best_prediction2data is None or prediction2data is None:
    #         continue
    #     for k in set(best_prediction2data.keys()):
    #         if prediction2data.get(k, {'dMRR': 0})['dMRR'] > best_prediction2data.get(k, {'dMRR': 0})['dMRR']:
    #             best_prediction2data[k] = prediction2data[k]

    # for setting in ['eXpath()1', 'data_poisoning', 'k1', 'KGEAttack']:
    #     prediction2data = loadAverageData(folder, setting)
    #     if best_prediction2data is None or prediction2data is None:
    #         continue
    #     for k in set(best_prediction2data.keys()):
    #         diff = 0.003 if setting == 'eXpath()1' else 0.001
    #         if prediction2data.get(k, {'dMRR': 0})['dMRR'] + diff >= best_prediction2data.get(k, {'dMRR': 0})['dMRR']:
    #             best_prediction2data[k] = prediction2data[k]
    #             best_count.loc[setting, folder] += 1


for folder in folders:
    process(folder)

# AVG = sum / notnull_count
rx['AVG'] = rx.mean(axis=1)
rx_h['AVG'] = rx_h.mean(axis=1)
rx_t['AVG'] = rx_t.mean(axis=1)
fx['AVG'] = fx.mean(axis=1)
DHit1['AVG'] = DHit1.mean(axis=1)
DHit1_h['AVG'] = DHit1_h.mean(axis=1)
DHit1_t['AVG'] = DHit1_t.mean(axis=1)
# gx['AVG'] = gx.mean(axis=1)
best_count['AVG'] = best_count.mean(axis=1)


base_dir='metric'
rx.to_csv(f'{base_dir}/rx.csv')
rx_h.to_csv(f'{base_dir}/rx_h.csv')
rx_t.to_csv(f'{base_dir}/rx_t.csv')
fx.to_csv(f'{base_dir}/fx.csv')
DHit1.to_csv(f'{base_dir}/DHit1.csv')
DHit1_h.to_csv(f'{base_dir}/DHit1_h.csv')
DHit1_t.to_csv(f'{base_dir}/DHit1_t.csv')
best_count.to_csv(f'{base_dir}/best_count.csv')
# gx.to_csv(f'{base_dir}/gx.csv')