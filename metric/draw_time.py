import matplotlib.pyplot as plt
import numpy as np

# Re-define data due to state reset
models = ['complex', 'conve', 'transe']
datasets = ['FB15k', 'WN18', 'FB15k-237', 'WN18RR']

kelpie_data = {
    'complex': [130, 28, 105, 30],
    'conve': [137.5, 34.7, 35, 111.1],
    'transe': [80, 50, 25, 30],
}

eXpath_data = {
    'complex': [36.34, 9.16, 28.99, 10.21],
    'conve': [78.32, 18.95, 33.31, 7.44],  # Updated eXpath conve_FB15k to 78.32
    'transe': [46.87, 14.8, 14.62, 8.33],
}

# Recalculate averages
kelpie_avg = {model: np.mean(data) for model, data in kelpie_data.items()}
eXpath_avg = {model: np.mean(data) for model, data in eXpath_data.items()}

# Adjusting the chart layout to be square
fig, ax = plt.subplots(figsize=(12, 12))  # Square figure size
bar_width = 0.45
plt.rcParams.update({'font.size': 26, 'font.family': 'Meera'})

# Offset for each model
model_offsets = np.arange(len(models)) * (len(datasets) + 1)

# Plot each model's data in separate groups
for i, model in enumerate(models):
    kelpie_values = kelpie_data[model]
    eXpath_values = eXpath_data[model]
    
    # Calculate positions for each dataset within a group
    y_positions = model_offsets[i] + np.arange(len(datasets))
    
    # Horizontal bars for Kelpie and eXpath
    ax.barh(y_positions - bar_width / 2, kelpie_values, bar_width, label='Kelpie', color='blue', alpha=0.7)
    ax.barh(y_positions + bar_width / 2, eXpath_values, bar_width, label='eXpath', color='darkorange', alpha=0.7)
    
    # Add average lines for each model, confined to group
    ax.plot([kelpie_avg[model], kelpie_avg[model]],
            [y_positions[0] - bar_width, y_positions[-1] + bar_width], color='blue', linestyle='--', linewidth=1)
    ax.plot([eXpath_avg[model], eXpath_avg[model]],
            [y_positions[0] - bar_width, y_positions[-1] + bar_width], color='darkorange', linestyle='--', linewidth=1)
    
    # Annotate average values
    ax.text(kelpie_avg[model] - 12, y_positions[-1] + bar_width, f'AVG: {kelpie_avg[model]:.2f} s', color='blue', va='center')
    ax.text(eXpath_avg[model] - 12, y_positions[-1] + bar_width, f'AVG: {eXpath_avg[model]:.2f} s', color='darkorange', va='center')

# Set y-axis ticks for all datasets within models
y_ticks = []
y_labels = []

for i, model in enumerate(models):
    y_ticks.extend(model_offsets[i] + np.arange(len(datasets)))
    y_labels.extend([f'{dataset}' for dataset in datasets])

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# Add a secondary y-axis for model labels
model_ticks = model_offsets + len(datasets) / 2 - 0.5
model_labels = [f'{model.capitalize()}' for model in models]
ax_secondary = ax.twinx()
ax_secondary.set_yticks(model_ticks)
ax_secondary.set_yticklabels(model_labels)
ax_secondary.set_ylim(ax.get_ylim())

# Add labels and title
ax.set_xlabel('Time (s)')
ax.set_title('Kelpie vs eXpath Time Comparison by Model and Dataset')
ax_secondary.set_ylabel('Models')
ax.grid(True, linestyle='-', color='gray', alpha=0.3) 
# Adjust font for grid labels

plt.tight_layout()

# Show the plot
plt.savefig('time.png')
