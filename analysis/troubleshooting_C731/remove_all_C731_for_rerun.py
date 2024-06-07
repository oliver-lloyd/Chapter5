import pandas as pd
from pathlib import Path
from os import listdir

problem_rel = 'C0000731'

for path in Path('..').rglob('polySE_results_aggregations.csv'):
    print(f'Preparing {path} for re-run')
    df = pd.read_csv(path)
    df.query(f'side_effect != "{problem_rel}"', inplace=True)
    df.to_csv(path, index=False)

for path in Path('..').rglob('polySE_posthoc_results_aggregations.csv'):
    print(f'Preparing {path} for re-run')
    df = pd.read_csv(path)
    df.query(f'side_effect != "{problem_rel}"', inplace=True)
    df.to_csv(path, index=False)
