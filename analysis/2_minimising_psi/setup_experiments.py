import pandas as pd
import yaml

# Get info of best performing neighbourhood
ranks = pd.read_csv('average_ranks.csv')
best = ranks.iloc[0][['neighbourhood', 'nearest_n']]
best_str = f'{best.neighbourhood}_n{best.nearest_n}'

# Load a corresponding config to use as a template
template_path = f'../1_general_assessment/experiments/{best.neighbourhood}/{best_str}/{best_str}_psi_0.125.yaml'
with open(template_path, 'r') as f:
    template_config = yaml.safe_load(f)

for i in range(1, 6):
    new_psi = 10**-i
    config = template_config.copy()
    config['lookup_embedder']['psi'] = new_psi
    with open(f'experiments/{best_str}_psi_{new_psi}.yaml', 'w') as f:
        yaml.safe_dump(config, f)
