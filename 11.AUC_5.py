import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import warnings
import re
import shap
from tqdm import tqdm
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import StratifiedKFold
sys.path.append("/home/user/wangrui/04.project/17.RCI/01.best_machine_learning_python/LGBM_liuguanghui/DementiaProteomicPrediction-main/")
from Utility.Training_Utilities import *
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV

rawdf = pd.read_csv('00.rci_data_with_protemics.csv', index_col = 0)

rawdf = rawdf.rename(columns={'region': 'Region_code', 'rci': 'target_y', 'rci_y_time': 'BL2Target_yrs'})

rawdf.loc[(rawdf['target_y'] == 1) & (rawdf['BL2Target_yrs'] > 5), 'target_y'] = 0

Y = 'target_y'

T = 'BL2Target_yrs'

features = pd.read_csv('08.ProImportance.csv')

X = features.head(n=10)['Pro_code'].tolist()

#### protein panel

X_data = rawdf[X]

y = rawdf[Y]

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_depth': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    'num_leaves': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'learning_rate': [0.00005, 0.0005, 0.005, 0.05, 0.1],
    'subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    'colsample_bytree': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
}

X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=2025, stratify=y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

model = LGBMClassifier(class_weight='balanced', random_state=2025, n_jobs=4)

search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, scoring='roc_auc', n_jobs=20, cv=cv, random_state=42, verbose=1)

search.fit(X_data, y)

best_model = search.best_estimator_

y_pred_proba = best_model.predict_proba(X__test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

roc_auc = auc(fpr, tpr)

#### traditional factors
lst = ['Age', 'Sex', 'Edu', 'BMI', 'TD', 'Ethnic', 'Smoke', 'Drink', 'Hypertension', 'Diabetes', 'Obesity', 'Dyslipidemia']

X_traditional_data = rawdf[lst]

X_traditional_train, X_traditional_test, y_train, y_test = train_test_split(X_traditional_data, y, test_size=0.2, random_state=2025, stratify=y)

search.fit(X_traditional_train, y_train)

best_traditional_model = search.best_estimator_

y_pred_proba_traditional = best_traditional_model.predict_proba(X_traditional_test)[:,1]

fpr_traditional, tpr_traditional, _ = roc_curve(y_test, y_pred_proba_traditional)

roc_auc_traditional = auc(fpr_traditional, tpr_traditional)

#### traditionals factors + protein panel
X_t_p_data = rawdf[features.head(n=10)['Pro_code'].tolist() + lst]

X_t_p_train, X_t_p_test, y_train, y_test = train_test_split(X_t_p_data, y, test_size=0.2, random_state=2025, stratify=y)

search.fit(X_t_p_data, y)

best_model_t_p = search.best_estimator_

y_pred_proba = best_model_t_p.predict_proba(X_t_p_test)[:,1]

fpr_t_p, tpr_t_p, thresholds_t_p = roc_curve(y_test, y_pred_proba)

roc_auc_t_p = auc(fpr_t_p, tpr_t_p)

#### plot
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, color='#e99452',
         label=f'Protein Panel (AUC = {roc_auc:.3f})')

plt.plot(fpr_traditional, tpr_traditional, color='#4888b3',
         label=f'Traditional Factors (AUC = {roc_auc_traditional:.3f})')

plt.plot(fpr_t_p, tpr_t_p, color='#e57d7e',
         label=f'Protein Panel + Traditional Factors(AUC = {roc_auc_t_p:.3f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: Protein Panel vs Traditional Factors')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('11.roc_curve_5_comparison.pdf', dpi=300)
plt.close()
