from rdkit import Chem
from tmap import Minhash
from map4 import MAP4Calculator
from itertools import combinations
import pandas as pd


smiles_df = pd.read_csv('../../../../Chapter4/data/raw/canonical_smiles.csv')

dim = 1024
MAP4 = MAP4Calculator(dimensions=dim)
mols = [Chem.MolFromSmiles(smile_str) for smile_str in smiles_df['canonical smiles']]
fingerprints = MAP4.calculate_many(mols)

ENC = Minhash(dim)
out = []
for i, j in combinations(smiles_df.index, 2):
    drug_i = smiles_df.drug[i]
    drug_j = smiles_df.drug[j]
    dist = ENC.get_distance(fingerprints[i], fingerprints[j])
    out.append([drug_i, drug_j, dist, 1-dist])

out_df = pd.DataFrame(out, columns=['drug1', 'drug2', 'minhash_distance', 'minhash_similarity'])
out_df.to_csv('MAP4_minhash.csv', index=False)
