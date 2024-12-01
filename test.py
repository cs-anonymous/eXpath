import sys, os, argparse, numpy, torch
from dataset import Dataset, ALL_DATASET_NAMES
from link_prediction.evaluation.evaluation import Evaluator
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.transe import TransE
from link_prediction.models.model import BATCH_SIZE, LEARNING_RATE, EPOCHS, DIMENSION, MARGIN, NEGATIVE_SAMPLES_RATIO, REGULARIZER_WEIGHT, INIT_SCALE, INPUT_DROPOUT, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, HIDDEN_LAYER_SIZE, LABEL_SMOOTHING, DECAY
import pandas as pd

sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

def initialize_model(args, dataset):
    if args.model == 'complex':
        return ComplEx(dataset=dataset, hyperparameters={DIMENSION: args.dimension, INIT_SCALE: args.init_scale}, init_random=True)
    elif args.model == 'conve':
        return ConvE(dataset=dataset, hyperparameters={DIMENSION: args.dimension, INPUT_DROPOUT: args.input_dropout, FEATURE_MAP_DROPOUT: args.feature_map_dropout, HIDDEN_DROPOUT: args.hidden_dropout, HIDDEN_LAYER_SIZE: args.hidden_size, BATCH_SIZE: args.batch_size, LEARNING_RATE: args.learning_rate, DECAY: args.decay_rate, LABEL_SMOOTHING: args.label_smoothing, EPOCHS: args.max_epochs}, init_random=True)
    elif args.model == 'transe':
        return TransE(dataset=dataset, hyperparameters={DIMENSION: args.dimension, MARGIN: args.margin, NEGATIVE_SAMPLES_RATIO: args.negative_samples_ratio, REGULARIZER_WEIGHT: args.regularizer_weight, BATCH_SIZE: args.batch_size, LEARNING_RATE: args.learning_rate, EPOCHS: args.max_epochs}, init_random=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=ALL_DATASET_NAMES, required=True, help="The dataset to use")
parser.add_argument("--model", type=str, choices=['complex', 'conve', 'transe'], required=True, help="The model to use")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model to load")
parser.add_argument("--dimension", type=int, default=200, help="Embedding dimension")
parser.add_argument("--init_scale", type=float, default=1e-3, help="Initial scale (only for ComplEx)")
parser.add_argument("--input_dropout", type=float, default=0.3, help="Input layer dropout (only for ConvE)")
parser.add_argument("--hidden_dropout", type=float, default=0.4, help="Dropout after the hidden layer (only for ConvE)")
parser.add_argument("--feature_map_dropout", type=float, default=0.5, help="Dropout after the convolutional layer (only for ConvE)")
parser.add_argument("--hidden_size", type=int, default=9728, help="Hidden layer size (only for ConvE)")
parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing (only for ConvE)")
parser.add_argument("--decay_rate", type=float, default=1.0, help="Decay rate (only for ConvE)")
parser.add_argument("--margin", type=int, default=5, help="Margin for pairwise ranking loss (only for TransE)")
parser.add_argument("--negative_samples_ratio", type=int, default=3, help="Number of negative samples (only for TransE)")
parser.add_argument("--regularizer_weight", type=float, default=0.0, help="Regularization weight (only for TransE)")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--max_epochs", type=int, default=1000, help="Number of epochs")
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

args.output_folder = f'out/{args.model}_{args.dataset}'
os.makedirs(args.output_folder, exist_ok=True)

def evaluate():
    print(f"Loading {args.dataset} dataset...")
    dataset = Dataset(name=args.dataset, separator="\t", load=True)

    print(f"Initializing {args.model} model...")
    model = initialize_model(args, dataset)
    model.to('cuda')
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    print("Evaluating model...")
    mrr, h1, h10, mr = Evaluator(model=model).evaluate(samples=dataset.test_samples, write_output=True)
    print(f"\tTest Hits@1: {h1:.6f}")
    print(f"\tTest Hits@10: {h10:.6f}")
    print(f"\tTest Mean Reciprocal Rank: {mrr:.6f}")
    print(f"\tTest Mean Rank: {mr:.6f}")

    # move filtered_ranks.csv, filtered_details.csv to output folder
    os.rename('filtered_ranks.csv', os.path.join(args.output_folder, 'filtered_ranks.csv'))
    os.rename('filtered_details.csv', os.path.join(args.output_folder, 'filtered_details.csv'))

evaluate()
print("Generating facts to explain (prediction that ranks first)...")
lis = []
print("{:^30}\t{:^15}\t{:^15}".format('relation', '#triples', '#top_triples'))
print(os.path.join(args.output_folder, 'filtered_ranks.csv'))
df = pd.read_csv(os.path.join(args.output_folder, 'filtered_ranks.csv'), sep=';', header=None, dtype=str)
df.columns = ['h', 'r', 't', 'hr', 'tr']
df['hr'] = df['hr'].astype(int)
df['tr'] = df['tr'].astype(int)

for d in set(df['r']):
    rel_df = df[df['r'] == d]
    rel_df.reset_index(inplace=True)
    top_count = 0
    for i in range(len(rel_df)):
        # if df.loc[i, 'tr'] <= math.ceil(size*0.05):
        if rel_df.loc[i, 'tr'] != 1:
            continue
        
        # make sure tr and hr are 1 except for MOF dataset
        top_count += 1
        lis.append('\t'.join([str(rel_df.loc[i, 'h']), rel_df.loc[i, 'r'], str(rel_df.loc[i, 't'])]))
    print("{:^30}\t{:^15}\t{:^15}".format(d, len(rel_df), top_count))

# choose all facts to explain in lis
with open(os.path.join(args.output_folder, 'input_facts.csv'), 'w') as f:
    f.write('\n'.join(lis))
print(lis)
