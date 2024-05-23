import pandas as pd
from torch import tensor, save
from itertools import combinations


def add_reverse_edges(df):
    df2 = df.copy()
    df2['drug1'] = df['drug2']
    df2['drug2'] = df['drug1']
    return pd.concat([df, df2])


if __name__ == '__main__':

    # Get drug indexer
    selfloops_ids = pd.read_csv('../../../kge/data/selfloops/entity_ids.del', sep='\t', header=None)
    name_to_id = {row[1]: row[0] for _, row in selfloops_ids.iterrows()}


    data = {
        'RDK': {},
        'MAP4': {},
        'morgan': {},
    }

    # Load RDK similarity data
    RDK_raw = pd.read_csv('../../../Chapter4/data/processed/drug_fingerprint_similarity.csv')
    data['RDK']['raw'] = add_reverse_edges(RDK_raw)  # we DO need to add reverse edges to all these dfs, despite what I keep believing and wasting time on
    data['RDK']['target_col'] = 'fingerprint_similarity'

    # Load MAP4 similarity data
    map4_raw = pd.read_csv('MAP4/MAP4_minhash.csv')
    data['MAP4']['raw'] = add_reverse_edges(map4_raw)
    data['MAP4']['target_col'] = 'minhash_similarity'

    # Load Morgan similarity data
    morgan_raw = pd.read_csv('morgan/morgan_tanimoto.csv')
    data['morgan']['raw'] = add_reverse_edges(morgan_raw)
    data['morgan']['target_col'] = 'morgan_tanimoto'

    neighbourless = []
    
    # Filter by n closest neighbours for each drug
    for n in [5, 10, 20]:
        nearest_n_str = f'nearest_{n}'
        for neighbour_method in data.keys():


            target_col = data[neighbour_method]['target_col']
            out = []
            for drug, subdf in data[neighbour_method]['raw'].groupby('drug1'):
                subdf.query(f'{target_col} > 0', inplace=True)
                n_neighbours = len(subdf)
                if n_neighbours < n:
                    neighbourless.append([neighbour_method, n, drug, n_neighbours])
                if n_neighbours > 1:    
                    subdf.sort_values(target_col, ascending=False, inplace=True)
                    out.append(subdf.iloc[:n])
            out = pd.concat(out)

            out['id1'] = [name_to_id[name] for name in out.drug1.values]
            out['id2'] = [name_to_id[name] for name in out.drug2.values]
            out_tensor = tensor(out[['id1', 'id2']].to_numpy())
            data[neighbour_method][nearest_n_str] = out_tensor

        # Assert equivalence of neighbourhood graphs
        for key, key2 in combinations(data.keys(), 2):
            pass#assert data[key][nearest_n_str].shape == data[key2][nearest_n_str].shape
        
        # Save
        for key in data.keys():
            save(data[key][nearest_n_str], f'{key}/{key}_{nearest_n_str}.pt')

    # Do stuff if found some neighbourless drugs
    if len(neighbourless):
        neighbourless = pd.DataFrame(neighbourless, columns=['similarity_method', 'nearest_N', 'drug', 'actual_N'])
        neighbourless.to_csv('neighbourless.csv', index=False)

    # Using drug-gene bipartite projection edges (no nearest-n performed)
    gene_bipartite = pd.read_csv('../../../Chapter4/data/processed/drug_projection_edges.csv')
    gene_bipartite = add_reverse_edges(gene_bipartite)
    gene_bipartite['id1'] = [name_to_id[name] for name in gene_bipartite.drug1.values]
    gene_bipartite['id2'] = [name_to_id[name] for name in gene_bipartite.drug2.values]
    gene_bipartite_out = tensor(gene_bipartite[['id1', 'id2']].to_numpy())
    save(gene_bipartite_out, 'drug_gene_projection/gene_projection_edgelist.pt')
