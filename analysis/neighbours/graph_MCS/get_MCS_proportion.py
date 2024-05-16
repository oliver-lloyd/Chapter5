from rdkit import Chem
from itertools import combinations
from funcs import *
import pandas as pd
import networkx as nx
import multiprocessing as mp


if __name__ == '__main__':
    
    # Load molecule graphs
    smiles_df = pd.read_csv('../../../../Chapter4/data/raw/canonical_smiles.csv')
    mols = [Chem.MolFromSmiles(smile_str) for smile_str in smiles_df['canonical smiles']]
    graphs = [molecule_to_graph(mol) for mol in mols]

    # Get pairwise proportion of MCS size vs size of smaller drug
    args = [[graphs, i, j] for i, j in combinations(smiles_df.index, 2)]
    args = args[:8]
    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.starmap(MCS_proportion, args)

    # Save
    out_df = pd.DataFrame(res, columns=['id1', 'id2', 'MCS_proportion'])
    out_df['drug1'] = [smiles_df.drug[i] for i in out_df.id1]
    out_df['drug2'] = [smiles_df.drug[i] for i in out_df.id2]
    out_df = out_df[['drug1', 'drug2', 'MCS_proportion']]
    out_df.to_csv('MCS_proportion.csv', index=False)
