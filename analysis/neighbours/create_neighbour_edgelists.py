import pandas as pd
from torch import tensor, save

def add_reverse_edges(df):
    df2 = df.copy()
    df2['drug1'] = df['drug2']
    df2['drug2'] = df['drug1']
    return pd.concat([df, df2])

selfloops_ids = pd.read_csv('../../../kge/data/selfloops/entity_ids.del', sep='\t', header=None)
name_to_id = {row[1]: row[0] for _, row in selfloops_ids.iterrows()}

drug_sim_raw = pd.read_csv('../../../Chapter4/data/processed/drug_fingerprint_similarity.csv')
drug_sim_raw = add_reverse_edges(drug_sim_raw)  # we DO need to add reverse edges here because similarity results go both ways which isnt represented in the csv. Filtered to closest ten anyway

for n in [5, 10, 20]:
    drug_sim = []
    for drug, subdf in drug_sim_raw.groupby('drug1'):
        subdf.sort_values('fingerprint_similarity', ascending=False, inplace=True)
        drug_sim.append(subdf.iloc[:n])
    drug_sim = pd.concat(drug_sim)

    drug_sim['id1'] = [name_to_id[name] for name in drug_sim.drug1.values]
    drug_sim['id2'] = [name_to_id[name] for name in drug_sim.drug2.values]
    drug_sim_out = tensor(drug_sim[['id1', 'id2']].to_numpy())
    save(drug_sim_out, f'drug_sim_edgelist_nearest{n}.pt')

gene_bipartite = pd.read_csv('../../../Chapter4/data/processed/drug_projection_edges.csv')
gene_bipartite = add_reverse_edges(gene_bipartite)
gene_bipartite['id1'] = [name_to_id[name] for name in gene_bipartite.drug1.values]
gene_bipartite['id2'] = [name_to_id[name] for name in gene_bipartite.drug2.values]
gene_bipartite_out = tensor(gene_bipartite[['id1', 'id2']].to_numpy())
save(gene_bipartite_out, 'gene_bipartite_edgelist.pt')
