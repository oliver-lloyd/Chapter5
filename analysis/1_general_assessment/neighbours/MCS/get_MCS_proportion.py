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

    # Check for partial results
    result_file = 'MCS_proportion.csv'
    try:
        with open(result_file, 'r') as f:
            lines = f.read().split('\n')[1:]
            if lines[-1] == '':
                lines = lines[:-1]
            completed = [[int(val) for val in l.split(',')[:-1]] for l in lines]
    except FileNotFoundError:
        with open(result_file, 'x') as f:
            f.write('drug1,drug2,MCS_proportion\n')
        completed = []

    # Get pairwise proportion of MCS size vs size of smaller drug
    args = [[graphs, i, j, result_file] for i, j in combinations(smiles_df.index, 2) if [i, j] not in completed]
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(MCS_proportion, args)
