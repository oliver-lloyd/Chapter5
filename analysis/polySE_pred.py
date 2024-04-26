import pandas as pd
import torch

from os import listdir
from sklearn.metrics import roc_auc_score, average_precision_score
from argparse import ArgumentParser
from dotenv import dotenv_values

from kge.model import KgeModel
from kge.util.io import load_checkpoint
from kge.model.simple import SimplEScorer

# Disable pd warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Get enviroment info
env = dotenv_values()
libkge_path = env['LIBKGE_PATH']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load decagon rank metrics
import sys
sys.path.append(env['THESIS_PATH'] + '/Chapter3/analysis/assessment')
import decagon_rank_metrics

# Get user args
parser = ArgumentParser()
parser.add_argument('drug_vectors_path')
args = parser.parse_args()

# Load drug vectors
learned_vecs_df = pd.read_csv(args.drug_vectors_path)
learned_vecs = {}
for i, row in learned_vecs_df.iterrows():
    drug = row.drug
    vec = row[row.index[1:]].astype(float).to_numpy()
    vec = torch.tensor(vec).to(device)
    learned_vecs[drug] = vec

# Load chapter 3 relation embeddings
checkpoint = load_checkpoint(env['THESIS_PATH'] + '/Chapter5/data/selfloops/checkpoint_best.pt')
checkpoint['config'].set('dataset.name', F'{libkge_path}/data/selfloops')
embeds = KgeModel.create_from(checkpoint).state_dict()
rel_embeds = embeds['_relation_embedder._embeddings.weight'].to(device)
ent_embeds = embeds['_entity_embedder._embeddings.weight'].to(device)

# Create index dict for relation names
rel_name_to_id = {}
with open(f'{libkge_path}/data/selfloops/relation_ids.del', 'r') as f:
    for line in f.readlines():
        if line.endswith('\n'):
            line = line[:-1]
        rel_id, rel_name = line.split('\t')
        rel_name_to_id[rel_name] = int(rel_id)

# Create index dict for entity names
ent_name_to_id = {}
with open(f'{libkge_path}/data/selfloops/entity_ids.del', 'r') as f:
    for line in f.readlines():
        if line.endswith('\n'):
            line = line[:-1]
        ent_id, ent_name = line.split('\t')
        ent_name_to_id[ent_name] = int(ent_id)

# Load partial results if they exist
results_path = 'polySE_results.csv'
try:
    results = pd.read_csv(results_path)
except FileNotFoundError:
    results = pd.DataFrame(columns=['side_effect', 'fake_triple_component', 'AUROC', 'AUPRC', 'AP50'])

# Iterate over holdout edges (by side effect) and score them
fake_holdout_path = env['THESIS_PATH'] + '/Chapter3/analysis/assessment/false_edges'
holdout_edges = pd.read_csv(env['THESIS_PATH'] + '/Chapter5/data/selfloops/holdout.tsv', header=None, sep='\t')
holdout_edges['is_real_edge'] = 1
scorer = SimplEScorer(checkpoint['config'], 'selfloops')
n_dim = ent_embeds.shape[-1]
for side_effect_name, real_edges in holdout_edges.groupby(1):
    if side_effect_name in results.side_effect.values:
        print(f'Result found for {side_effect_name}, skipping..')
    else:
        print(f'Processing {side_effect_name}')
        
    rel_embed = rel_embeds[rel_name_to_id[side_effect_name]].view(1, n_dim)
    if side_effect_name == 'C0000731':
        print(rel_embed)

    # Create edge df to score
    false_edges = pd.read_csv(f'{fake_holdout_path}/{side_effect_name}.tsv', header=None, sep='\t')
    false_edges['is_real_edge'] = 0
    holdout_to_score = pd.concat([real_edges, false_edges])
    holdout_to_score['head_replaced_score'] = None
    holdout_to_score['tail_replaced_score'] = None
    holdout_to_score['both_replaced_score'] = None
    
    # Score all edges
    for i, row in holdout_to_score.iterrows():
        # Load head node info and vectors
        head_drug = row[0]
        head_drug_id = ent_name_to_id[head_drug]

        head_agg = learned_vecs[head_drug].view(1, n_dim)
        head_embed = ent_embeds[head_drug_id].view(1, n_dim)
        
        # Same for tail node
        tail_drug = row[2]
        tail_drug_id = ent_name_to_id[tail_drug]

        tail_agg = learned_vecs[tail_drug].view(1, n_dim)
        tail_embed = ent_embeds[tail_drug_id].view(1, n_dim)

        # Score three ways, replacing one or both components with the learned aggregation each time
        head_score = scorer.score_emb(head_agg, rel_embed, tail_embed, combine='spo').item()
        holdout_to_score['head_replaced_score'][i] = head_score

        tail_score = scorer.score_emb(head_embed, rel_embed, tail_agg, combine='spo').item()
        holdout_to_score['tail_replaced_score'][i] = tail_score

        both_score = scorer.score_emb(head_agg, rel_embed, tail_agg, combine='spo').item()
        holdout_to_score['both_replaced_score'][i] = both_score

    # Save calculated scores in case needed for future analysis
    holdout_to_score.to_csv(f'holdout_scores/{side_effect_name}.csv', index=False)

    # Assess the outcome scores using usual metrics
    # Assess once per each variation of replaced triple components
    for component in ['head', 'tail','both']:
        target_column = f'{component}_replaced_score'

        # Get area-under metrics
        labels = holdout_to_score.is_real_edge.values
        preds = holdout_to_score[target_column].values
        roc = roc_auc_score(labels, preds)
        prc = average_precision_score(labels, preds)

        # Get ap50
        holdout_to_score.sort_values(target_column, ascending=False, inplace=True)
        pos_index = holdout_to_score.query('is_real_edge == 1').index.values.tolist()
        sorted_index = holdout_to_score.index.values.tolist()
        ap50 = decagon_rank_metrics.apk(
            pos_index,
            sorted_index,
            k=50
        )

        # Store outcome
        results.loc[len(results)] = [side_effect_name, component, roc, prc, ap50]
    
    # Save partial results after every side effect
    results.to_csv(results_path, index=False)
    