import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed, cpu_count
import pymannkendall as mk
import warnings
warnings.filterwarnings('ignore')

scaler = MinMaxScaler()

protein_list = pd.read_csv('08.ProImportance.csv')
protein_list = protein_list['Pro_code'].head(10).tolist()

rawdf = pd.read_csv('00.data.csv')
rawdf = rawdf[rawdf['rci'].isin([0, 1])].copy()
rawdf['CaseFlag'] = rawdf['rci'].astype(int)
rawdf = rawdf[(rawdf.rci_y_time <= 14) & (rawdf.rci_y_time > 1)]

covariates = ['eid', 'Age', 'Sex', 'Edu', 'rci', 'rci_y_time'] + protein_list
df_psm = rawdf[covariates].dropna(subset=covariates).copy()

df_psm_encoded = pd.get_dummies(df_psm, columns=['Sex', 'Edu'], drop_first=True)

X = df_psm_encoded.drop(columns=['eid', 'rci'] + protein_list)
y = df_psm_encoded['rci']
logit = LogisticRegression(max_iter=1000)
logit.fit(X, y)
df_psm['PropensityScore'] = logit.predict_proba(X)[:, 1]

cases = df_psm[df_psm['rci'] == 1].copy()
controls = df_psm[df_psm['rci'] == 0].copy()
match_number = 5

nn = NearestNeighbors(n_neighbors=match_number, metric='euclidean')
nn.fit(controls[['PropensityScore']])

def match_case(case_row):
    case_id = case_row['eid']
    case_score = np.array([[case_row['PropensityScore']]])
    distances, indices = nn.kneighbors(case_score)
    matched_ctrls = controls.iloc[indices[0]]['eid'].values
    return [{'CaseID': case_id, 'ControlID': ctrl_id} for ctrl_id in matched_ctrls]

results = Parallel(n_jobs=min(cpu_count(), 40), backend='loky')(
    delayed(match_case)(row) for _, row in cases.iterrows()
)

matched_pairs = [pair for sublist in results for pair in sublist]
matched_df = pd.DataFrame(matched_pairs)

print(f"Number of matched case: {matched_df['CaseID'].nunique()} / {len(cases)}")
print(f"Number of matched control: {matched_df['ControlID'].nunique()} / {len(controls)}")
print(f"Number of matched people: {len(matched_df)} (每个病例约 {len(matched_df)/len(cases):.1f} 个对照)")

cases_matched = cases[cases.eid.isin(matched_df['CaseID'].unique())]
controls_matched = controls[controls.eid.isin(matched_df['ControlID'].unique())]
df_matched = pd.concat([cases_matched, controls_matched], axis=0, ignore_index=True)

for col in protein_list:
    if col in df_matched.columns:
        df_matched[col] = scaler.fit_transform(df_matched[[col]])

df_long = pd.melt(
    df_matched,
    id_vars=['eid', 'rci', 'rci_y_time'],
    value_vars=protein_list,
    var_name='Protein',
    value_name='Level'
)

df_long['Time_bin'] = np.round(df_long['rci_y_time']).astype(int) - 15

trend_results = []
for protein in protein_list:
    df_prot = df_long[df_long['Protein'] == protein].copy()
    plt.figure(figsize=(6, 4))
    for label, color in zip([1, 0], ['#c25e64', '#23b6ea']):
        df_group = df_prot[df_prot['rci'] == label]
        grouped = df_group.groupby('Time_bin')['Level']
        mean_vals = grouped.mean()
        sem_vals = grouped.sem()
        x = mean_vals.index.values
        y = mean_vals.values
        ci95 = 1.96 * sem_vals.values
        if len(x) > 3:
            spline_mean = UnivariateSpline(x, y, s=0.5)
            spline_upper = UnivariateSpline(x, y + ci95, s=0.5)
            spline_lower = UnivariateSpline(x, y - ci95, s=0.5)
            x_smooth = np.linspace(x.min(), x.max(), 200)
            plt.plot(x_smooth, spline_mean(x_smooth), color=color,
                     label='Case' if label == 1 else 'Control', linewidth=2)
            plt.fill_between(x_smooth, spline_lower(x_smooth), spline_upper(x_smooth),
                             color=color, alpha=0.2)
        else:
            plt.plot(x, y, color=color, label='Case' if label == 1 else 'Control', linewidth=2)
            plt.fill_between(x, y - ci95, y + ci95, color=color, alpha=0.2)
        mk_result = mk.original_test(y)
        trend_results.append({
            'Protein': protein,
            'Group': 'Case' if label == 1 else 'Control',
            'Trend': mk_result.trend,
            'p_value': mk_result.p,
            'Sen_slope': mk_result.slope
        })
    plt.xlabel('Time to diagnosis (years)')
    plt.ylabel(f'Plasma {protein}')
    plt.legend()
    plt.xlim(-14, -1)
    plt.xticks(np.arange(-14, 0, 1))  
    plt.tight_layout()
    plt.savefig(f"13.plasma_trajectory_spline/protein_trajectory_{protein}_spline_PSM_parallel_{match_number}.png", dpi=300)
    plt.savefig(f"13.plasma_trajectory_spline/protein_trajectory_{protein}_spline_PSM_parallel_{match_number}.pdf", dpi=300)
    plt.close()

trend_df = pd.DataFrame(trend_results)
trend_df.to_csv(f"16.protein_trend_results_spline_PSM_parallel_{match_number}.csv", index=False)
