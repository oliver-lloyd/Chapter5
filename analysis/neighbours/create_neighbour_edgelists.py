import pandas as pd
from torch import tensor, save


def add_reverse_edges(df):
    df2 = df.copy()
    df2['drug1'] = df['drug2']
    df2['drug2'] = df['drug1']
    return pd.concat([df, df2])


if __name__ == '__main__':

    # Get drug indexer
    selfloops_ids = pd.read_csv('../../../kge/data/selfloops/entity_ids.del', sep='\t', header=None)
    name_to_id = {row[1]: row[0] for _, row in selfloops_ids.iterrows()}

    # Load RDK similarity data
    RDK_raw = pd.read_csv('../../../Chapter4/data/processed/drug_fingerprint_similarity.csv')
    RDK_raw = add_reverse_edges(RDK_raw)  # we DO need to add reverse edges here because similarity results go both ways which isnt represented in the csv. Filtered to closest ten anyway

    # Load MAP4 similarity data
    map4_raw = pd.read_csv('MAP4/MAP4_minhash.csv')
    map4_raw = add_reverse_edges(map4_raw)

    # Load Morgan similarity data
    morgan_raw = pd.read_csv('morgan_tanimoto/morgan_tanimoto.csv')
    morgan_raw = add_reverse_edges(morgan_raw)

    # Filter by n closest neighbours for each drug
    for n in [5, 10, 20]:

        # Using tanimoto coefficient of RDK fingerprints
        RDK = []
        for drug, subdf in RDK_raw.groupby('drug1'):
            subdf.sort_values('fingerprint_similarity', ascending=False, inplace=True)
            RDK.append(subdf.iloc[:n])
        RDK = pd.concat(RDK)

        RDK['id1'] = [name_to_id[name] for name in RDK.drug1.values]
        RDK['id2'] = [name_to_id[name] for name in RDK.drug2.values]
        RDK_out = tensor(RDK[['id1', 'id2']].to_numpy())
        

        # Using MAP4 minhash
        map4 = []
        for drug, subdf in map4_raw.groupby('drug1'):
            subdf.sort_values('minhash_similarity', ascending=False, inplace=True)
            map4.append(subdf.iloc[:n])
        map4 = pd.concat(map4)

        map4['id1'] = [name_to_id[name] for name in map4.drug1.values]
        map4['id2'] = [name_to_id[name] for name in map4.drug2.values]
        map4_out = tensor(map4[['id1', 'id2']].to_numpy())

        # Using Morgan tanimoto
        morgan = []
        for drug, subdf in morgan_raw.groupby('drug1'):
            subdf.sort_values('morgan_tanimoto', ascending=False, inplace=True)
            morgan.append(subdf.iloc[:n])
        morgan = pd.concat(morgan)

        morgan['id1'] = [name_to_id[name] for name in morgan.drug1.values]
        morgan['id2'] = [name_to_id[name] for name in morgan.drug2.values]
        morgan_out = tensor(morgan[['id1', 'id2']].to_numpy())

        # Assert equivalence then save
        assert map4_out.shape == RDK_out.shape
        assert morgan_out.shape == RDK_out.shape
        save(RDK_out, f'RDK_tanimoto/RDK_nearest{n}.pt')
        save(map4_out, f'MAP4/MAP4_nearest{n}.pt')
        save(morgan_out, f'morgan_tanimoto/morgan_nearest{n}.pt')

    # Using drug-gene bipartite projection edges (no nearest-n performed)
    gene_bipartite = pd.read_csv('../../../Chapter4/data/processed/drug_projection_edges.csv')
    gene_bipartite = add_reverse_edges(gene_bipartite)
    gene_bipartite['id1'] = [name_to_id[name] for name in gene_bipartite.drug1.values]
    gene_bipartite['id2'] = [name_to_id[name] for name in gene_bipartite.drug2.values]
    gene_bipartite_out = tensor(gene_bipartite[['id1', 'id2']].to_numpy())
    save(gene_bipartite_out, 'drug_gene_projection/gene_projection_edgelist.pt')
