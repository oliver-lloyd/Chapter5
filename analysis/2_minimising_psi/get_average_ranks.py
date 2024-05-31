import pandas as pd
import numpy as np

from os import listdir


def load_results(result_filename, exp_path='../1_general_assessment/experiments', target_len=963*3):
    out = []
    for loc1 in ['MAP4', 'RDK', 'morgan']:
        neighbour_method = loc1
        for loc2 in listdir(f'{exp_path}/{loc1}'):
            nearest_n = int(loc2.split('_n')[-1])
            for loc3 in listdir(f'{exp_path}/{loc1}/{loc2}'):
                if loc3.startswith('2024'):
                    load_path = f'{exp_path}/{loc1}/{loc2}/{loc3}/{result_filename}'
                    df = pd.read_csv(load_path)
                    if len(df) > target_len:
                        df = df.drop_duplicates()
                        assert len(df) == target_len
                        df.to_csv(load_path)
                    df['psi'] = float(loc3.split('psi_')[-1])
                    df['neighbourhood'] = neighbour_method if neighbour_method != "morgan" else "Morgan"
                    df['nearest_n'] = nearest_n
                    df['Training'] = 'OOS-KGE'
                    out.append(df)
    return out

# Load all PPSE results from part 1
chap5 = load_results('polySE_results_aggregations.csv')
chap5 = pd.concat(chap5)

# Get median performance over all side effects for each combination of variables
target_cols = ['neighbourhood', 'nearest_n', 'fake_triple_component']
summary_df = pd.DataFrame(columns= target_cols + ['median_AUPRC'])
for tup, subdf in chap5.groupby(target_cols):
    med_auprc = list(tup) + [subdf.AUPRC.median()]
    summary_df.loc[len(summary_df)] = med_auprc

# Order the combinations by those medians and write average rank statistics to file
summary_df = summary_df.sort_values('median_AUPRC', ascending=False).reset_index(drop=True)
out_df = pd.DataFrame(columns=target_cols[:-1] + ['mean_rank'])
for tup, subdf in summary_df.groupby(['neighbourhood', 'nearest_n']):
    out_row = list(tup) + [np.mean(subdf.index)]
    out_df.loc[len(out_df)] = out_row

# Sort and save
target_col = 'Mean rank'
out_df.columns = ['Neighbourhood (N)', 'Number of neighbours (n)', target_col]
out_df.sort_values(target_col, inplace=True)
out_df[target_col] = [f'{val:.3f}' for val in out_df[target_col].values]  # Convert to str to quickly trim trailing zeros
out_df.to_csv('average_ranks.csv', index=False)
with open('average_ranks.tex', 'w') as f:
    f.write(out_df.to_latex(index=False))
