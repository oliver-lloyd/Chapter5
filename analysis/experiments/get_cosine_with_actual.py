import pandas as pd
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from dotenv import dotenv_values
from scipy.spatial.distance import cosine


def cosine_sim(vec1, vec2):
    return 1 - cosine(vec1, vec2)


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

# Load aggregations 
agg_df = pd.read_csv('drug_sim_centroids.csv')
agg_df['neighbourhood_method'] = 'drug similarity'
agg_df['vector_method'] = 'mean components'
embed_cols = [str(i) for i in range(256)]

# Create result dataframe
cosine_df = pd.DataFrame(
    columns=['drug', 'neighbourhood_method', 'vector_method', 'cosine_to_actual']
)

# Calculate cosines
for i, row in agg_df.iterrows():
    aggregation = row[embed_cols].to_numpy()
    real_embed = node_embeds[row.selfloops_id].numpy()
    cos = cosine_sim(aggregation, real_embed)
    cosine_df.loc[len(cosine_df)] = [row.drug, row.neighbourhood_method, row.vector_method, cos]

# Save result
cosine_df.to_csv('cosines_vs_actual.csv', index=False)
