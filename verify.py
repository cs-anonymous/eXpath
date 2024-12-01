import sys
import os
import argparse
import copy
import random
import numpy
import torch
import json
import numpy as np
import time

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

import yaml
from link_prediction.evaluation.evaluation import Evaluator
from dataset import ALL_DATASET_NAMES, Dataset, MANY_TO_ONE, ONE_TO_ONE
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.transe import TransE
from link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from link_prediction.optimization.bce_optimizer import BCEOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from link_prediction.models.model import DIMENSION, INIT_SCALE, LEARNING_RATE, OPTIMIZER_NAME, DECAY_1, DECAY_2, \
    REGULARIZER_WEIGHT, EPOCHS, BATCH_SIZE, INPUT_DROPOUT, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, \
    HIDDEN_LAYER_SIZE, LABEL_SMOOTHING, DECAY, MARGIN, NEGATIVE_SAMPLES_RATIO, REGULARIZER_NAME

# Define available models
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

# Add optimizer choices
OPTIMIZER_CHOICES = ['Adagrad', 'Adam', 'SGD']
FILTER_CHOICES = ['head', 'tail', 'ht', 'none', 'greater']

parser = argparse.ArgumentParser(description="Model-agnostic tool for verifying link predictions explanations")

parser.add_argument('--model', choices=MODEL_CHOICES, required=True, help="Model to use: complex, conve, transe")
parser.add_argument('--dataset', choices=ALL_DATASET_NAMES, required=True, help="Dataset to use")
parser.add_argument('--optimizer', choices=OPTIMIZER_CHOICES, default='Adagrad', help="Optimizer to use")
parser.add_argument('--mode', type=str, default="necessary", choices=["sufficient", "necessary"], help="The explanation mode (default is necessary)")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--max_epochs', type=int, default=1000, help="Number of epochs")
parser.add_argument('--dimension', type=int, default=200, help="Factorization rank or embedding dimension")
parser.add_argument('--learning_rate', type=float, default=0.0005, help="Learning rate")
parser.add_argument('--reg', type=float, default=0, help="Regularization weight")
parser.add_argument('--init', type=float, default=1e-3, help="Initial scale for complex")
parser.add_argument('--decay_rate', type=float, default=0, help="Decay rate for Adagrad")
parser.add_argument('--decay1', type=float, default=0.9, help="Decay rate for the first moment estimate in Adam")
parser.add_argument('--decay2', type=float, default=0.999, help="Decay rate for second moment estimate in Adam")
parser.add_argument('--regularizer_name', type=str, default='N3', help="Regularizer name")
parser.add_argument('--margin', type=int, default=5, help="Margin for pairwise ranking loss (TransE)")
parser.add_argument('--negative_samples_ratio', type=int, default=3, help="Negative samples ratio (TransE)")
parser.add_argument('--input_dropout', type=float, default=0.3, help="Input layer dropout (ConvE)")
parser.add_argument('--hidden_dropout', type=float, default=0.4, help="Hidden layer dropout (ConvE)")
parser.add_argument('--feature_map_dropout', type=float, default=0.5, help="Feature map dropout (ConvE)")
parser.add_argument('--hidden_size', type=int, default=9728, help="Hidden layer size (ConvE)")
parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing (ConvE)")
parser.add_argument('--metric', type=str, default='GA')
parser.add_argument('--topN', type=int, default=4)
parser.add_argument('--filter', type=str, choices=FILTER_CHOICES, default='none')
parser.add_argument('--regularizer_weight', type=float, default=0.0, help="Regularizer weight")
parser.add_argument('--ablation', type=str, default='1111', help="Ablation study: CP, Uc, Ud, EXP")


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

# Read explanations from the output file
ablation_suffix = '_' + args.ablation if  args.ablation != '1111' else ''
map_path = f'out/{args.model}_{args.dataset}/extractedFactsMap{ablation_suffix}.json'
if args.metric in ['kelpie', 'criage', 'data_poisoning', 'k1', 'AnyBurlAttack']:
    dirname = os.path.join('results', 'necessary', args.metric, 
                           args.dataset.lower().replace('3-10', '').replace('-', ''), args.model.lower())
    if args.metric == 'kelpie':
        dirname = os.path.join(dirname, f"threshold_5")
    if not os.path.exists(f"{dirname}/output.txt"):
        print(f"Output file not found: {dirname}/output.txt")
        sys.exit(1)

    with open(f"{dirname}/output.txt", "r") as input_file:
        input_lines = input_file.readlines()

    data = []
    for i in range(0, len(input_lines), 3):
        fact_line = input_lines[i]
        rules_line = input_lines[i + 1]
        try:
            empty_line = input_lines[i + 2]
        except:
            print('empty line not found', dirname, i)

        prediction = fact_line.strip().strip(';').split(";")
        rule_relevance_inputs = rules_line.strip().split(",")
        best_rule, best_rule_relevance_str = rule_relevance_inputs[0].split(":")
        best_rule = best_rule.split(";")
        best_rule_relevance_str=best_rule_relevance_str.strip('[').strip(']')

        # print('processing:', prediction, best_rule, best_rule_relevance_str)
        data.append({
            "prediction": prediction,
            "explanation": [{
                "perspective": "head",
                "triples": [','.join([best_rule[ix], best_rule[ix+1], best_rule[ix+2]]) for ix in range(0, len(best_rule), 3)],
                "relevance": float(best_rule_relevance_str)
            }]
        })

elif os.path.exists(map_path):
    topN = args.topN
    metric = args.metric if args.metric != 'eXpath' else 'score'
    with open(map_path, "r") as input_file:
        extractedFactsMap = json.load(input_file)
    
    if args.filter != 'none':
        removed_keys = []
        for k, v in extractedFactsMap.items():
            if len(k.split(',')) != 3:
                print('invalid fact:', k)
                removed_keys.append(k)
                continue

            h = k.split(',')[0]
            t = k.split(',')[2]
            headCount = len([x for x in v if h in x['triple']])
            tailCount = len([x for x in v if t in x['triple']])
            if args.filter == 'head' or (args.filter =='greater' and headCount > tailCount):
                extractedFactsMap[k] = [x for x in v if h in x['triple']]
            elif args.filter == 'tail' or (args.filter =='greater' and tailCount > headCount):
                extractedFactsMap[k] = [x for x in v if t in x['triple']]
            elif args.filter == 'ht':
                extractedFactsMap[k] = [x for x in v if h in x['triple'] or t in x['triple']]

        for k in removed_keys:
            extractedFactsMap.pop(k)

    data = []
    for fact, explanation in extractedFactsMap.items():
        if '+' in metric:
            triples = []
            for m in metric.split('+'):
                explanation = sorted(explanation, key=lambda x: x[m], reverse=True)
                triples += [t['triple'] for t in explanation[:topN]]
            data.append({
                "prediction": fact.split(","),
                "explanation": [{
                    'triples': triples,
                    'relevance': np.mean([t[m] for t in explanation[:topN]])
                }]
            })
            
        else:
            print('processing:', fact, 'total length:', len(explanation))
            # asssert explanation is a list
            assert isinstance(explanation, list)
            explanation = sorted(explanation, key=lambda x: x[metric], reverse=True)
            data.append({
                "prediction": fact.split(","),
                "explanation": [{
                    'triples': [t['triple'] for t in explanation[:topN]],
                    'relevance': np.mean([t[metric] for t in explanation[:topN]])
                }]
            })
elif os.path.exists('output.json'):
    with open("output.json", "r") as input_file:
        data = json.load(input_file)

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
model.load_state_dict(torch.load(args.model_path))
model.eval()

facts_to_explain = []
samples_to_explain = []
perspective = "head"
sample_to_explain_2_best_rule = {}

# Explanation verification based on mode
if args.mode != "necessary":
    raise NotImplementedError("Only necessary mode is supported")
# Process necessary mode
for rule_relevance_inputs in data:
    rules_with_relevance = []
    fact = tuple(rule_relevance_inputs['prediction'])
    facts_to_explain.append(fact)
    sample = (dataset.entity_name_2_id[fact[0]], dataset.relation_name_2_id[fact[1]], dataset.entity_name_2_id[fact[2]])
    samples_to_explain.append(sample)

    best_rule_samples = []
    for x in rule_relevance_inputs['explanation'][0]['triples']:
        try:
            best_rule_samples.append(dataset.fact_to_sample(x.split(',')))
        except:
            print('invalid triple:', x)
    relevance = rule_relevance_inputs['explanation'][0]['relevance']
    rules_with_relevance.append((best_rule_samples, relevance))
    sample_to_explain_2_best_rule[sample] = best_rule_samples

samples_to_remove = []

for sample_to_explain in samples_to_explain:
    best_rule_samples = sample_to_explain_2_best_rule[sample_to_explain]
    samples_to_remove += best_rule_samples

new_dataset = copy.deepcopy(dataset)

print("Removing samples: ", len(samples_to_remove))
for (head, relation, tail) in samples_to_remove:
    print("\t" + dataset.printable_sample((head, relation, tail)))

new_dataset.remove_training_samples(numpy.array(samples_to_remove))

original_scores, original_ranks, original_predictions = model.predict_samples(numpy.array(samples_to_explain))

new_model = model_class(dataset=new_dataset, hyperparameters=hyperparameters, init_random=True)
new_optimizer = optimizer_class(model=new_model, hyperparameters=hyperparameters)
# print(new_dataset.train_samples)
new_optimizer.train(train_samples=new_dataset.train_samples)
new_model.eval()

new_scores, new_ranks, new_predictions = new_model.predict_samples(numpy.array(samples_to_explain))

output_lines = []
for i in range(len(samples_to_explain)):
    cur_sample_to_explain = samples_to_explain[i]

    data[i]['original'] = {
        "rank_head": original_ranks[i][0],          # head rank
        "rank_tail": original_ranks[i][1],          # tail rank
        "score_head": original_scores[i][0],        # head score
        "score_tail": original_scores[i][1],        # tail score
        "MRR_head": 1 / original_ranks[i][0],
        "MRR_tail": 1 / original_ranks[i][1],
        'MRR': (1 / original_ranks[i][0] + 1 / original_ranks[i][1]) / 2
    }
    data[i]['new'] = {
        "rank_head": new_ranks[i][0],          # head rank
        "rank_tail": new_ranks[i][1],          # tail rank
        "score_head": new_scores[i][0],        # head score
        "score_tail": new_scores[i][1],        # tail score
        "MRR_head": 1 / new_ranks[i][0],
        "MRR_tail": 1 / new_ranks[i][1],
        'MRR': (1 / new_ranks[i][0] + 1 / new_ranks[i][1]) / 2
    }
    data[i]['dMRR'] = data[i]['original']['MRR'] - data[i]['new']['MRR']

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


print('Required time: ', time.time() - t, ' seconds')
mrr, h1, h10, mr = Evaluator(model=model).evaluate(samples=dataset.test_samples, write_output=False)
new_mrr, new_h1, new_h10, new_mr = Evaluator(model=new_model).evaluate(samples=dataset.test_samples, write_output=False)
print({
    'mrr': mrr,
    'h1': h1,
    'h10': h10,
    'mr': mr,
    'new_mrr': new_mrr,
    'new_h1': new_h1,
    'new_h10': new_h10,
    'new_mr': new_mr
})

suffix = ''
if args.filter != 'none':
    suffix += args.filter[0]
if args.ablation != '1111':
    suffix += args.ablation
if args.metric == 'eXpath':
    suffix = '(' + suffix + ')'
filename = f"out/{args.model}_{args.dataset}/output_end_to_end_{args.metric}{suffix}{args.topN}.json"
print('output file:', filename)

with open(filename, "w") as outfile:
    json.dump(data, outfile, indent=4, cls=NumpyEncoder)