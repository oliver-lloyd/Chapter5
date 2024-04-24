import pandas as pd
from torch import save, tensor
from torch_geometric.data import Data


# Load edge data
raw_edgelist = pd.read_csv('../../../../Chapter4/data/processed/drug_projection_edges.csv')

# Add opposite direction edges
raw_edgelist2 = raw_edgelist.copy()
raw_edgelist2['drug1'] = raw_edgelist['drug2']
raw_edgelist2['drug2'] = raw_edgelist['drug1']
raw_edgelist = pd.concat([raw_edgelist, raw_edgelist2])

# No need to filter here, as was done for drug_similarity, as this graph is already sparse
""" 
# Filter for top k closest neighbours
k = 10
edgelist = []
for drug, subdf in raw_edgelist.groupby('drug1'):
    subdf.sort_values('normalised_weight', inplace=True, ascending=False)
    topk = subdf.iloc[:k]
    edgelist.append(topk)
edgelist = pd.concat(edgelist)
"""
edgelist = raw_edgelist

# Index nodes
drugs = set(edgelist.drug1.unique()).union(set(edgelist.drug2.unique()))
drug_index = {drug: i for i, drug in enumerate(drugs)}
drug_index = drug_index | {drug_index[key]: key for key in drugs}  # Merge with inverted drug index for either-way indexing

# Convert drug name to node id
edgelist['drug1_id'] = [drug_index[s] for s in edgelist.drug1.values]
edgelist['drug2_id'] = [drug_index[s] for s in edgelist.drug2.values]

# Load drug embeds
drug_embeds = pd.read_csv('../../../data/drug_embeds.csv')
drug_embeds.query('name in @drugs', inplace=True)
drug_embeds['id'] = [drug_index[name] for name in drug_embeds.name.values]
drug_embeds.sort_values('id', inplace=True)

# Instantiate graph with embeds as node features AND node targets
edge_index = tensor(edgelist[['drug1_id', 'drug2_id']].to_numpy()).T
drug_features = tensor(drug_embeds[[str(i) for i in range(256)]].to_numpy()).float()
target_drug_features = drug_features.clone()  # Clone, otherwise can't modify x without changing y
data = Data(
    x=drug_features, 
    y=target_drug_features, 
    edge_index=edge_index, 
    edge_weight=edgelist.normalised_weight.to_numpy()
)

# Add index for converting node IDs back to drug names
data.drug_index = drug_index

# Save
save(data, 'drug_projection_pyg.pt')
