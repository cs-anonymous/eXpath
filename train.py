import sys
import os
import argparse
import random
import time
import numpy
import torch
import json
import numpy as np

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

import yaml
from dataset import ALL_DATASET_NAMES, Dataset
from link_prediction.evaluation.evaluation import Evaluator
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.transe import TransE
from link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from link_prediction.optimization.bce_optimizer import BCEOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from link_prediction.models.model import DIMENSION, INIT_SCALE, LEARNING_RATE, OPTIMIZER_NAME, DECAY_1, DECAY_2, \
    REGULARIZER_WEIGHT, EPOCHS, BATCH_SIZE, REGULARIZER_NAME, INPUT_DROPOUT, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, \
    HIDDEN_LAYER_SIZE, LABEL_SMOOTHING, DECAY, MARGIN, NEGATIVE_SAMPLES_RATIO
from prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER

# Define model choices
MODEL_CHOICES = ['complex', 'conve', 'transe']

def read_yaml(file_path):
    """
    Read a YAML file and parse it into a dictionary.
    
    :param file_path: str, the path to the YAML file
    :return: dict, the parsed YAML content as a dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    for k, v in config.items():
        if 'reg' in v:
            v['reg'] = float(v['reg'])
    return config

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")
parser.add_argument('--model', choices=MODEL_CHOICES, required=True, help="Model to use: complex, conve, transe")
parser.add_argument('--dataset', choices=ALL_DATASET_NAMES, required=True, help="Dataset to use")
parser.add_argument('--optimizer', choices=['Adagrad', 'Adam', 'SGD'], default='Adagrad', help="Optimizer to use")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--max_epochs', type=int, default=200, help="Number of epochs")
parser.add_argument('--dimension', type=int, default=200, help="Factorization rank or embedding dimension")
parser.add_argument('--learning_rate', type=float, default=1e-1, help="Learning rate")
parser.add_argument('--reg', type=float, default=0, help="Regularization weight")
parser.add_argument('--init', type=float, default=1e-3, help="Initial scale for complex")
parser.add_argument('--decay_rate', type=float, default=0, help="Decay rate for Adagrad")
parser.add_argument('--decay1', type=float, default=0.9, help="Decay rate for the first moment estimate in Adam")
parser.add_argument('--decay2', type=float, default=0.999, help="Decay rate for second moment estimate in Adam")
parser.add_argument('--margin', type=int, default=5, help="Margin for pairwise ranking loss (TransE)")
parser.add_argument('--negative_samples_ratio', type=int, default=3, help="Negative samples ratio (TransE)")
parser.add_argument('--input_dropout', type=float, default=0.3, help="Input layer dropout (ConvE)")
parser.add_argument('--hidden_dropout', type=float, default=0.4, help="Hidden layer dropout (ConvE)")
parser.add_argument('--feature_map_dropout', type=float, default=0.5, help="Feature map dropout (ConvE)")
parser.add_argument('--hidden_size', type=int, default=9728, help="Hidden layer size (ConvE)")
parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing (ConvE)")
parser.add_argument('--coverage', type=int, default=10, help="Number of random entities to extract and convert")
parser.add_argument('--baseline', type=str, default=None, help="Baseline engine to use")
parser.add_argument('--entities_to_convert', type=str, help="Path of the file with the entities to convert (baselines)")
parser.add_argument('--relevance_threshold', type=float, default=None, help="Relevance acceptance threshold")
parser.add_argument('--prefilter', choices=[TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER], default=NO_PREFILTER, help="Prefilter type")
parser.add_argument('--prefilter_threshold', type=int, default=20, help="Number of promising training facts to keep after prefiltering")
parser.add_argument('--relation', type=str, help="Relation to explain")
parser.add_argument('--perspective', type=str, default="head", choices=["head", "tail", "double"], help="The perspective to explain")
parser.add_argument('--regularizer_weight', type=float, default=0.0, help="Regularizer weight")
parser.add_argument('--valid', default=-1, type=float, help="Number of epochs before valid.")

args = parser.parse_args()
config = read_yaml("config.yaml")
for k, v in config[f'{args.model}_{args.dataset}'].items():
    setattr(args, k, v)  # Use setattr to add/modify attributes in args

print('args', args)

# Set random seed for reproducibility
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

# Load dataset
print(f"Loading dataset {args.dataset}...")
dataset = Dataset(name=args.dataset, separator="\t", load=True)

# Define model-specific parameters
if args.model == 'complex':
    hyperparameters = {DIMENSION: args.dimension, INIT_SCALE: args.init, LEARNING_RATE: args.learning_rate,
                       OPTIMIZER_NAME: args.optimizer, DECAY_1: args.decay1, DECAY_2: args.decay2, 
                       REGULARIZER_WEIGHT: args.reg, EPOCHS: args.max_epochs, BATCH_SIZE: args.batch_size,
                       REGULARIZER_NAME: "N3"}
    model_class = ComplEx
    optimizer_class = MultiClassNLLOptimizer

elif args.model == 'conve':
    hyperparameters = {DIMENSION: args.dimension, INPUT_DROPOUT: args.input_dropout, FEATURE_MAP_DROPOUT: args.feature_map_dropout,
                       HIDDEN_DROPOUT: args.hidden_dropout, HIDDEN_LAYER_SIZE: args.hidden_size, BATCH_SIZE: args.batch_size,
                       LEARNING_RATE: args.learning_rate, DECAY: args.decay1, LABEL_SMOOTHING: args.label_smoothing, 
                       EPOCHS: args.max_epochs, OPTIMIZER_NAME: args.optimizer}
    model_class = ConvE
    optimizer_class = BCEOptimizer

elif args.model == 'transe':
    hyperparameters = {DIMENSION: args.dimension, MARGIN: args.margin, NEGATIVE_SAMPLES_RATIO: args.negative_samples_ratio,
                       REGULARIZER_WEIGHT: args.regularizer_weight, BATCH_SIZE: args.batch_size, LEARNING_RATE: args.learning_rate, 
                       EPOCHS: args.max_epochs, OPTIMIZER_NAME: args.optimizer}
    model_class = TransE
    optimizer_class = PairwiseRankingOptimizer

# Initialize and load the model
t = time.time()
model = model_class(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
model.to('cuda')
# model.load_state_dict(torch.load(args.model_path))
# model.eval()

optimizer = optimizer_class(model=model, hyperparameters=hyperparameters)
optimizer.train(train_samples=dataset.train_samples,
                save_path=args.model_path,
                evaluate_every=args.valid,
                valid_samples=dataset.valid_samples)

print("Evaluating model...")
mrr, h1, h10, mr = Evaluator(model=model).evaluate(samples=dataset.test_samples, write_output=False)
print("\tTest Hits@1: %f" % h1)
print("\tTest Hits@10: %f" % h10)
print("\tTest Mean Reciprocal Rank: %f" % mrr)
print("\tTest Mean Rank: %f" % mr)