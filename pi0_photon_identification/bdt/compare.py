# -------------------------------------------------------------------------------------------
#
# compare.py
#
# Compare AUC and ROC of BDT trained inclusively vs combination of 5 BDTs trained
# separately in non-overlapping cluster energy ranges
#
# Author: Giovanni Marchiori (giovanni.marchiori@cern.ch)
#
# -------------------------------------------------------------------------------------------


import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import roc_curve as rc
import pandas as pd
import matplotlib.pyplot as plt

bdt = {}
X_test = {}
y_test = {}
y_test_pred = {}
outdirs = ['inclusive', '0-20', '20-40', '40-60', '60-80', '80-100']

for outdir in outdirs:
    # Load the arrays from the .npz file
    data = np.load('./' + outdir + '/models/bdt-inputs-train-test-split.npz', allow_pickle=True)
    X_train = data['X_train']
    X_test[outdir] = data['X_test']
    y_train = data['y_train']
    y_test[outdir] = data['y_test']
    column_names = data['columns']
    bdt[outdir] = lgb.Booster(model_file='./' + outdir + '/models/bdt-photonid.txt')
    y_test_pred[outdir] = bdt[outdir].predict(X_test[outdir])

combined_array = np.column_stack((y_test['inclusive'], y_test_pred['inclusive']))
df_inclusive = pd.DataFrame(combined_array)
df_inclusive.columns = ['class', 'prediction']
print(df_inclusive)

test_auc = {}
for outdir in outdirs:
    test_auc[outdir] = ras(y_test[outdir], y_test_pred[outdir])

y_test_5bdts = np.concatenate([y_test['0-20'],
                               y_test['20-40'],
                               y_test['40-60'],
                               y_test['60-80'],
                               y_test['80-100']])
y_test_pred_5bdts = np.concatenate([y_test_pred['0-20'],
                                    y_test_pred['20-40'],
                                    y_test_pred['40-60'],
                                    y_test_pred['60-80'],
                                    y_test_pred['80-100']])

combined_array = np.column_stack((y_test_5bdts, y_test_pred_5bdts))
df_5bdts = pd.DataFrame(combined_array)
df_5bdts.columns = ['class', 'prediction']
print(df_5bdts)
test_auc_5bdts = ras(y_test_5bdts, y_test_pred_5bdts)

print("ROC-AUC for inclusive training:", test_auc['inclusive'] )
print("ROC-AUC for 5 BDT training:", test_auc_5bdts)

# now draw the various ROCs
fpr = {}
tpr = {}
for outdir in outdirs:
    fpr[outdir], tpr[outdir], _ = rc(y_test[outdir], y_test_pred[outdir])
fpr_5bdts, tpr_5bdts, _ = rc(y_test_5bdts, y_test_pred_5bdts)


one_minus_fpr = {}
for outdir in outdirs:
    one_minus_fpr[outdir] = 1 - fpr[outdir]
one_minus_fpr_5bdts = 1 - fpr_5bdts

# Plot  ROC curve with TPR on x-axis (eff_gamma) and 1 - FPR (1-eff_pi0) on y-axis
fig = plt.figure()
plt.plot(tpr['inclusive'], one_minus_fpr['inclusive'], color='blue', lw=2, label='ROC curve - inclusive training (AUC=%.3f)' % test_auc['inclusive'])
plt.plot(tpr_5bdts, one_minus_fpr_5bdts, color='orange', lw=2, label='ROC curve - 5 BDT training (AUC=%.3f)' % test_auc_5bdts)
plt.plot([0, 1], [1, 0], color='red', lw=2, linestyle='--', label='Random classifier (AUC=0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('$\\varepsilon_{\\gamma}$')
plt.ylabel('$1-\\varepsilon_{\\pi^{0}}$')
plt.title('ROC Curve')
plt.legend(loc="lower left")
plt.grid(True)
# plt.show()
# plt.plot(fpr, tpr, label="data, auc="+str(test_auc))
# plt.legend(loc=4)
plt.savefig('inclusive/plots/bdt-photonid-roc-comparison.pdf')
plt.close(fig)

fig = plt.figure()
for outdir in outdirs:
    plt.plot(tpr[outdir], one_minus_fpr[outdir], lw=2, label='%s (AUC=%.3f)' % (outdir, test_auc[outdir]))
# plt.plot([0, 1], [1, 0], color='red', lw=2, linestyle='--', label='Random classifier (AUC=0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('$\\varepsilon_{\\gamma}$')
plt.ylabel('$1-\\varepsilon_{\\pi^{0}}$')
plt.title('ROC Curve')
plt.legend(loc="lower left")
plt.grid(True)
# plt.show()
# plt.plot(fpr, tpr, label="data, auc="+str(test_auc))
# plt.legend(loc=4)
plt.savefig('inclusive/plots/bdt-photonid-roc-comparison2.pdf')
plt.close(fig)
