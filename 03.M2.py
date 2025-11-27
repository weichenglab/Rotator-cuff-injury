import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
pd.options.mode.chained_assignment = None  # default='warn'
from joblib import Parallel, delayed
from statsmodels.stats.multitest import fdrcorrection, multipletests as bonferroni_correction
from tqdm import tqdm

rawdf = pd.read_csv('00.data.csv')

rawdf = rawdf.rename(columns={'region': 'Region_code', 'rci': 'target_y', 'rci_y_time': 'BL2Target_yrs'})

pro_df = rawdf[['eid'] + rawdf.columns[37:2960].tolist()]
pro_f_lst = rawdf.columns[37:2960].tolist()
pro_dict = df_pro = pd.DataFrame({"Pro_code": pro_f_lst, "Pro_definition": pro_f_lst})

cov_df = rawdf[rawdf.columns[0:37].tolist()]

m1_f_lst = ['Age', 'Sex', 'Edu']
m2_f_lst = m1_f_lst + ['BMI', 'TD', 'Ethnic', 'Smoke', 'Drink', 'Hypertension', 'Diabetes', 'Obesity', 'Dyslipidemia']


target_df = rawdf[['eid', 'target_y', 'BL2Target_yrs']]
target_df.BL2Target_yrs.describe()

mydf = rawdf

myout_df, pro_out_lst = pd.DataFrame(), []
i=0

def fit_single_protein_m2(mydf, pro_f, m2_f_lst):
    tmpdf_f = ['target_y', 'BL2Target_yrs', pro_f] + m2_f_lst
    tmpdf = mydf[tmpdf_f].copy()
    tmpdf.rename(columns={pro_f: "target_pro"}, inplace=True)
    tmpdf.dropna(subset=["target_pro"], inplace=True)
    tmpdf.reset_index(drop=True, inplace=True)
    cph = CoxPHFitter()
    my_formula = "age+sex+edu+bmi+TD_index+ethnic+smoke+drink+hypertension+diabetes+target_pro"
    try:
        cph.fit(tmpdf, duration_col='BL2Target_yrs', event_col='target_y', formula=my_formula)
        hr = cph.hazard_ratios_["target_pro"]
        ci = np.exp(cph.confidence_intervals_.loc["target_pro"])
        pval = cph.summary.loc["target_pro", "p"]
        return (pro_f, hr, ci.iloc[0], ci.iloc[1], pval)
    except Exception as e:
        print(f"Error in {pro_f}: {e}")
        return None

results = Parallel(n_jobs=80)(
    delayed(fit_single_protein_m2)(mydf, pro_f, m2_f_lst) for pro_f in tqdm(pro_f_lst)
)

results = [r for r in results if r is not None]

myout_df = pd.DataFrame(results, columns=['Pro_code', 'HR', 'HR_Lower_CI', 'HR_Upper_CI', 'HR_p_val'])

_, p_f_fdr = fdrcorrection(myout_df.HR_p_val.fillna(1))
_, p_f_bfi, _, _ = bonferroni_correction(myout_df.HR_p_val.fillna(1), alpha=0.05)

myout_df['p_val_fdr'] = p_f_fdr
myout_df['p_val_bfi'] = p_f_bfi

myout_df.to_csv('03.Cox_M2_all_new.csv', index=False)
