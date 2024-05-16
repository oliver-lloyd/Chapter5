from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from itertools import combinations
import pandas as pd


# Load molecules from canonical SMILES
smiles_df = pd.read_csv('../../../../Chapter4/data/raw/canonical_smiles.csv')
mols = [Chem.MolFromSmiles(smile_str) for smile_str in smiles_df['canonical smiles']]

# Get Morgan fingerprints
radius = 3  # default value
morgan_gen = AllChem.GetMorganGenerator(radius)
morgan_fprints = [morgan_gen.GetSparseCountFingerprint(mol) for mol in mols]

# Get pairwise Tanimoto similarity
out = []
for i, j in combinations(smiles_df.index, 2):
    drug_i = smiles_df.drug[i]
    drug_j = smiles_df.drug[j]
    sim = TanimotoSimilarity(morgan_fprints[i], morgan_fprints[j])
    out.append([drug_i, drug_j, sim])

# Save
out_df = pd.DataFrame(out, columns=['drug1', 'drug2', 'morgan_tanimoto'])
out_df.to_csv('morgan_tanimoto.csv', index=False)
