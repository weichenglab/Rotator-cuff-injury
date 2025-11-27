import re
import sys
import numpy as np
import pandas as pd
sys.path.append("/home/wulab/wangrui/03.project/01.CVD/00.yujintai/DementiaProteomicPrediction-main/")
from Utility.Training_Utilities import *
import lightgbm as lgb
import warnings
from tqdm import tqdm
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut

rawdf = pd.read_csv('00.data.csv')

rawdf = rawdf.rename(columns={'region': 'Region_code', 'rci': 'target_y', 'rci_y_time': 'BL2Target_yrs'})

pro_df = rawdf[['eid'] + rawdf.columns[29:2952].tolist()]
pro_f_lst = rawdf.columns[29:2952].tolist()
pro_dict = df_pro = pd.DataFrame({"Pro_code": pro_f_lst, "Pro_definition": pro_f_lst})

cov_df = rawdf[rawdf.columns[0:29].tolist()]

m1_f_lst = ['Age', 'Sex', 'Edu']
m2_f_lst = m1_f_lst + ['BMI', 'TD', 'Ethnic', 'Smoke', 'Drink', 'Hypertension', 'Diabetes', 'Obesity', 'Dyslipidemia']

target_df = rawdf[['eid', 'target_y', 'BL2Target_yrs']]
target_df.BL2Target_yrs.describe()

rawdf['stratify_col'] = rawdf['target_y'].astype(str) + '_' + rawdf['Region_code'].astype(str)
rawdf['in_cv_fold'] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

for fold, (_, val_idx) in enumerate(skf.split(rawdf, rawdf['stratify_col'])):
    rawdf.loc[val_idx, 'in_cv_fold'] = fold

rawdf.drop(columns=['stratify_col'], inplace=True)

mydf = rawdf

fold_id_lst = list(set(mydf.Region_code))
fold_id_lst = [int(ele) for ele in fold_id_lst]
inner_cv_fold_lst = list(set(mydf.in_cv_fold))
inner_cv_fold_lst = [int(ele) for ele in inner_cv_fold_lst]

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

param_dist = {
    "n_estimators": list(range(100, 1001, 100)),
    "num_leaves": list(range(10, 101, 10)),
    "max_depth": list(range(3, 31, 3)),
    "subsample": np.round(np.arange(0.7, 1.01, 0.05), 2).tolist(),
    "colsample_bytree": np.round(np.arange(0.7, 1.01, 0.05), 2).tolist(),
    "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
}

model = lgb.LGBMClassifier(class_weight='balanced', random_state=2025, n_jobs=4)

pro_f_m1_df = pd.read_csv('./02.Cox_M1_all.csv')

pro_f_m2_df = pd.read_csv('./03.Cox_M2_all.csv')

pro_f_m1 = pro_f_m1_df.loc[pro_f_m1_df.p_val_bfi < 0.05].Pro_code.tolist()

pro_f_m2 = pro_f_m2_df.loc[pro_f_m2_df.p_val_bfi < 0.05].Pro_code.tolist()

pro_f_lst = [ele for ele in pro_f_m2 if ele in pro_f_m1]

X = pro_df[pro_f_lst]
y = rawdf.target_y
groups = mydf["Region_code"]

train_idx = []
test_idx = []

for region, idx in mydf.groupby("Region_code").groups.items():
    idx = np.array(list(idx))
    tr, te = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=y.iloc[idx]
    )
    train_idx.extend(tr)
    test_idx.extend(te)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

logo = LeaveOneGroupOut()

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=1000,
    scoring="roc_auc",
    random_state=2025,
    cv=logo,
    n_jobs=20
)

random_search.fit(X_train, y_train, groups=groups_train)

my_lgb = random_search.best_estimator_

tg_imp_cv = Counter()

totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
feature_names = my_lgb.booster_.feature_name()

totalgain_imp = dict(zip(feature_names, totalgain_imp))

tg_imp_cv += Counter(totalgain_imp)

tg_imp_df = pd.DataFrame({
    'Pro_code': list(tg_imp_cv.keys()),
    'TotalGain_cv': list(tg_imp_cv.values())
})

if tg_imp_df['TotalGain_cv'].sum() > 0:
    tg_imp_df['TotalGain_cv'] = tg_imp_df['TotalGain_cv'] / tg_imp_df['TotalGain_cv'].sum()

my_imp_df = pd.merge(tg_imp_df, pro_dict, how='left', on='Pro_code')

my_imp_df.sort_values(by='TotalGain_cv', ascending=False, inplace=True)

my_imp_df.to_csv('08.ProImportance.csv')
