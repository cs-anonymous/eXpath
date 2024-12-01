import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据
models = ['ComplEx', 'ConvE', 'TransE']
settings = ['eXpath', 'DP', 'Kelpie', 'AnyBurl']
datasets = ['FB15k', 'FB15k-237', 'WN18', 'WN18RR']

data = {
    'FB15k': {
        'ComplEx': [32, 34, 50, 37],
        'ConvE': [32, 38, 39, 44],
        'TransE': [43, 40, 35, 46]
    },
    'FB15k-237': {
        'ComplEx': [40, 27, 42, 33],
        'ConvE': [34, 22, 32, 26],
        'TransE': [33, 44, 29, 25]
    },
    'WN18': {
        'ComplEx': [60, 54, 32, 54],
        'ConvE': [77, 57, 43, 77],
        'TransE': [74, 66, 53, 74]
    },
    'WN18RR': {
        'ComplEx': [49, 50, 41, 55],
        'ConvE': [54, 59, 42, 64],
        'TransE': [33, 50, 62, 59]
    }
}

plt.rcParams.update({'font.size': 20, 'font.family': 'Meera'})

df = pd.read_csv('best_count.csv')
for dataset in datasets:
    data[dataset] = {}
    for model in models:
        data[dataset][model] = [df.loc[index, f'{model.lower()}_{dataset}'] for index in df.index]

# 创建2x2的图表
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 为每个数据集绘制图表
for i, dataset in enumerate(datasets):
    ax = axes[i // 2, i % 2]  # 获取子图
    width = 0.2  # 柱子的宽度
    x = np.arange(len(models))  # 每个模型的x轴位置

    # 为每个设置绘制柱状图
    for j, setting in enumerate(settings):
        # 获取每个模型的高度数据
        heights = [data[dataset][model][j] for model in models]
        
        # 绘制柱状图
        bars = ax.bar(x + j * width, heights, width, label=setting)
        
        # 在每个柱子底部添加百分比标识
        for ix, bar in enumerate(bars):
            height = bar.get_height()  # 获取柱子的高度
            # 计算百分比
            # percentage = height / sum(heights) * 100
            # 在柱子底部添加文本，逆时针旋转90°
            ax.text(bar.get_x() + bar.get_width() / 2, 2, heights[ix], 
                    ha='center', va='bottom', color='white', rotation=0, fontsize=24)
    
    # 设置图表标题、标签和格式
    ax.set_title(f'{dataset}')
    ax.set_xticks(x + width * (len(settings) - 1) / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel('#Best Explanation')
    ax.legend()

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('best_count.png')