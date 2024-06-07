import pandas as pd
import torch

from os import listdir
from sklearn.metrics import roc_auc_score, average_precision_score
from argparse import ArgumentParser
from dotenv import dotenv_values

from kge.model import KgeModel
from kge.util.io import load_checkpoint
from kge.model.simple import SimplEScorer

# Get user args
parser = ArgumentParser()
parser.add_argument('--posthoc_removal', action='store_true', default=False)
args = parser.parse_args()

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

# Load drug vectors
learned_vecs_df = pd.read_csv('drug_sim_centroids.csv').drop(columns=['selfloops_id'])
learned_vecs = {}
for i, row in learned_vecs_df.iterrows():
    drug = row.drug
    vec = row[row.index[1:]].astype(float).to_numpy()
    vec = torch.tensor(vec).to(device)
    learned_vecs[drug] = vec

# Load chapter 3 relation embeddings
try:
    checkpoint = load_checkpoint('checkpoint_best.pt')
except OSError:
    from os import listdir
    checkpoint_files = [f for f in listdir() if f.endswith('.pt')]
    n_checkpoints = len(checkpoint_files)
    if n_checkpoints == 1:
        checkpoint = load_checkpoint(checkpoint_files[0])
    else:
        raise FileNotFoundError(f'Couldn\'t find checkpoint_best, then found {n_checkpoints} other checkpoints. Please remove all except target checkpoint.')
checkpoint['config'].set('dataset.name', F'{libkge_path}/data/selfloops')
embeds = KgeModel.create_from(checkpoint).state_dict()
rel_embeds = embeds['_relation_embedder._embeddings.weight'].to(device)

ent_embeds = embeds['_entity_embedder._embeddings.weight'].to(device)
n_dim = ent_embeds.shape[-1]

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
results_path = 'polySE_results_aggregations.csv' if not args.posthoc_removal else 'polySE_posthoc_results_aggregations.csv'
try:
    results = pd.read_csv(results_path)
except FileNotFoundError:
    results = pd.DataFrame(columns=['side_effect', 'fake_triple_component', 'AUROC', 'AUPRC', 'AP50'])



# Load holdout data
fake_holdout_path = env['THESIS_PATH'] + '/Chapter3/analysis/assessment/false_edges'
holdout_edges = pd.read_csv(env['THESIS_PATH'] + '/Chapter5/data/selfloops/holdout.tsv', header=None, sep='\t')
holdout_edges.columns=['drug1', 'side_effect', 'drug2']
holdout_edges['is_real_edge'] = 1

# Load problem nodes if required
if args.posthoc_removal:
    neighbourless = pd.read_csv(f'{env["THESIS_PATH"]}/Chapter5/analysis/neighbours/neighbourless.csv')
    ignore_nodes = neighbourless.drug.unique()
    holdout_edges.query('drug1 not in @ignore_nodes and drug2 not in @ignore_nodes', inplace=True)

""" This is currently not necessary as only doing OOS learning with drug-sim network.
Same goes for the two 'false_edges.query()' lines in the loop below.

# Filter holdout edges if any nodes not in learned vectors (e.g. drug-gene projection only has half drugs)
old_len = len(holdout_edges)
drugs = list(learned_vecs.keys())
holdout_edges.query('drug1 in @drugs', inplace=True)
holdout_edges.query('drug2 in @drugs', inplace=True)
new_len = len(holdout_edges)
if new_len < old_len:
    print(f'Warning: removed {old_len - new_len} holdout edges that contained nodes with no learned vectors. ')
"""

# Iterate over holdout edges (grouped by side effect) and score them
scorer = SimplEScorer(checkpoint['config'], 'selfloops')
for side_effect_name, real_edges in holdout_edges.groupby('side_effect'):
    if side_effect_name in results.side_effect.values:
        print(f'Result found for {side_effect_name}, skipping..')
    else:
        print(f'Processing {side_effect_name}')
        
        rel_embed = rel_embeds[rel_name_to_id[side_effect_name]].view(1, n_dim)

        # Load fake holdout edges and filter
        false_edges = pd.read_csv(f'{fake_holdout_path}/{side_effect_name}.tsv', header=None, sep='\t')
        false_edges.columns=['drug1', 'side_effect', 'drug2']
        if args.posthoc_removal:
            false_edges.query('drug1 not in @ignore_nodes and drug2 not in @ignore_nodes', inplace=True)
        false_edges['is_real_edge'] = 0

        # Combine real with fake holdout edges and add placeholder score columns
        holdout_to_score = pd.concat([real_edges, false_edges]).reset_index(drop=True)
        holdout_to_score['head_replaced_score'] = None
        holdout_to_score['tail_replaced_score'] = None
        holdout_to_score['both_replaced_score'] = None
        
        # Score all edges
        for i, row in holdout_to_score.iterrows():
            # Load head node info and vectors
            head_drug = row['drug1']
            head_drug_id = ent_name_to_id[head_drug]

            head_agg = learned_vecs[head_drug].view(1, n_dim)
            head_embed = ent_embeds[head_drug_id].view(1, n_dim)
            
            # Same for tail node
            tail_drug = row['drug2']
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
        if not args.posthoc_removal:
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
        