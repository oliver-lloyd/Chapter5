import pandas as pd
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from dotenv import dotenv_values

# Set paths
libkge_path = dotenv_values()['LIBKGE_PATH']
selfloops_data_path = libkge_path + '/data/selfloops'

# Load node info
selfloops_ids = pd.read_csv(selfloops_data_path + '/entity_ids.del', sep='\t', header=None)
node_indexer = {row[1]: row[0] for _, row in selfloops_ids.iterrows()}
node_indexer.update({node_indexer[key]: key for key in node_indexer})

# Load model
checkpoint = load_checkpoint('checkpoint_best.pt')
checkpoint['config'].set('dataset.name', selfloops_data_path)
model = KgeModel.create_from(checkpoint)
node_embeds = model.state_dict()['_entity_embedder._embeddings.weight']
neighbour_adj = model._entity_embedder.neighbour_adj

# Construct output df
outfile_name = 'drug_sim_centroids.csv'
try:
    out = pd.read_csv(outfile_name)
except FileNotFoundError:
    out = pd.DataFrame(columns = ['drug', 'selfloops_id'] + [str(i) for i in range(node_embeds.shape[-1])])

# Go
for node_id, neighbours_sparse in enumerate(neighbour_adj):
    neighbour_ids = neighbours_sparse._indices()
    if neighbours_sparse._values().sum():  # Only drug nodes have neighbours here, skip the others
        if node_id not in out.selfloops_id.values:  # Also skip if already done
            print(node_id)

            # Double check this is a drug node
            drug_name = node_indexer[node_id]
            assert drug_name.startswith('CID')  

            # Get centroid of neighbours' embeddings
            neighbour_vecs = node_embeds[neighbour_ids][0]
            centroid = neighbour_vecs.mean(dim=0)

            # Save
            out_row = [drug_name, node_id] + centroid.tolist()
            out.loc[len(out)] = out_row
            out.to_csv(outfile_name, index=False)
