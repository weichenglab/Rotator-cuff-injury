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
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

rawdf = pd.read_csv('00.rci_data_with_protemics.csv', index_col = 0)

rawdf = rawdf.rename(columns={'region': 'Region_code', 'rci': 'target_y', 'rci_y_time': 'BL2Target_yrs'})

Y = 'target_y'

T = 'BL2Target_yrs'

features = pd.read_csv('03.ProImportance.csv')

X = features.head(n=10)['Pro_code'].tolist()

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

model = lgb.LGBMClassifier(class_weight='balanced', random_state=2025, n_jobs=4)

search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, scoring='roc_auc', n_jobs=20, cv=cv, random_state=42, verbose=1)

search.fit(X_tarin, y_train)

best_model = search.best_estimator_

explainer = shap.Explainer(best_model, X_test)
shap_values = explainer(X_test, check_additivity=False)

shap.summary_plot(shap_values, X_test, plot_type="dot", show=False, max_display=50)

fig = plt.gcf()
fig.set_size_inches(12, 8)

plt.tight_layout()
plt.savefig("03.shap_beeswarm_all.png", dpi=300)
plt.savefig("03.shap_beeswarm_all.pdf", dpi=300)
plt.close()

shap_df = pd.DataFrame(shap_values.values, columns=X_data.columns)
shap_df.to_csv("03.shap_values.csv", index=False)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_data, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig("03.shap_summary_bar.png", dpi=300)
plt.savefig("03.shap_summary_bar.pdf", dpi=300)
plt.close()

sample_index = np.argmax(best_model.predict_proba(X_data)[:, 1])
sample = X_data.iloc[sample_index]
sample_df = sample.to_frame().T
shap_values_sample = explainer(sample_df)
shap.plots.waterfall(shap_values_sample[0], show=False, max_display=20)
plt.tight_layout()
plt.savefig("03.shap_waterfall_case.png", dpi=300)
plt.savefig("03.shap_waterfall_case.pdf", dpi=300)
plt.close()

for feature in X[:10]:
    plt.figure(figsize=(6, 5))
    shap.dependence_plot(feature, shap_values.values, X_data, show=False)
    plt.tight_layout()
    plt.savefig(f"03.shap_dependence_{feature}.png", dpi=300)
    plt.savefig(f"03.shap_dependence_{feature}.pdf", dpi=300)
    plt.close()

