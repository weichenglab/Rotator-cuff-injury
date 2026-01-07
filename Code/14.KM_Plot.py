import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
pd.options.mode.chained_assignment = None
from lifelines import KaplanMeierFitter as kmf
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, statistics
from sklearn.metrics import roc_curve
from lifelines.plotting import add_at_risk_counts
from matplotlib.ticker import FuncFormatter

rawdf = pd.read_csv('00.data.csv', index_col = 0)

Y = 'rci'

T = 'rci_y_time'

features = pd.read_csv('08.ProImportance.csv')

X = features['Pro_code'].head(10).tolist()

outdir = "14.KM_results"
os.makedirs(outdir, exist_ok=True)

for f in X:
    tmp = rawdf[[T, Y, f]].dropna()
    fpr, tpr, thresholds = roc_curve(tmp[Y], tmp[f])
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_cutoff = thresholds[best_idx]
    best_J = youden_index[best_idx]
    high_risk = tmp[f] > best_cutoff
    low_risk = ~high_risk
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(10, 4))
    kmf_low.fit(tmp[T][low_risk], tmp[Y][low_risk], label="Low " + f)
    kmf_low.plot_survival_function(ax=ax, ci_show=False)
    kmf_high.fit(tmp[T][high_risk], tmp[Y][high_risk], label="High " + f)
    kmf_high.plot_survival_function(ax=ax, ci_show=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
    results = statistics.logrank_test(
        tmp[T][low_risk], tmp[T][high_risk],
        event_observed_A=tmp[Y][low_risk],
        event_observed_B=tmp[Y][high_risk])
    ax.set_title(f"(P={results.p_value:.3e})")
    ax.set_xlabel("Follow-up time (years)")
    ax.set_ylabel("Survival probability")
    add_at_risk_counts(kmf_low, kmf_high, ax=ax)
    outfile = os.path.join(outdir, f"KM_{f}_new.pdf")
    plt.savefig(outfile, bbox_inches="tight")
    outfile = os.path.join(outdir, f"KM_{f}_new.png")
    plt.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
