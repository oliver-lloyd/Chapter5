import pandas as pd
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from dotenv import dotenv_values
env = dotenv_values('../.env')
selfloops_path = env['LibKGE_path'] + '/data/selfloops'

# Load embeddings
checkpoint_path = 'selfloops/checkpoint_best.pt'
checkpoint = load_checkpoint(checkpoint_path)
checkpoint['config'].set('dataset.name', selfloops_path)
model = KgeModel.create_from(checkpoint)
all_embeds = model.state_dict()['_entity_embedder._embeddings.weight']

# Load node index
node_index = pd.read_csv(selfloops_path + '/entity_ids.del', header=None, sep='\t')
assert all(node_index[0] == node_index.index)  # Drop repeated index
node_index.drop(columns=0, inplace=True)
node_index.columns=['node_name']
node_index.index.name = 'selfloops_index'

# Get drug embeds
drugs = node_index.query('node_name.str.startswith("CID")')
drug_embeds = pd.DataFrame(all_embeds[drugs.index], index=drugs.index)
drug_embeds = pd.merge(drugs, drug_embeds, left_index=True, right_index=True)
drug_embeds.to_csv('drug_embeds.csv')


# TODO: get gene embeds for bipartite GNN
genes = []
for name in node_index.node_name.values:
    try:
        int(name)
        genes.append(name)
    except ValueError:
        continue
genes = node_index.query('node_name in @genes')
gene_embeds = pd.DataFrame(all_embeds[genes.index], index=genes.index)
gene_embeds = pd.merge(genes, gene_embeds, left_index=True, right_index=True)
gene_embeds.to_csv('gene_embeds.csv')