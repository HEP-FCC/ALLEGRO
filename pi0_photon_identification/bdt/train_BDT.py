# -------------------------------------------------------------------------------------------
#
# train_BDT.py
#
# Train and test a BDT for photon ID discrimination with the ALLEGRO ECAL
#
# Author: Giovanni Marchiori (giovanni.marchiori@cern.ch)
#
# -------------------------------------------------------------------------------------------

# imports

import argparse
import subprocess
import os
import math
import datetime
import uproot
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import onnx
import onnxmltools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import roc_curve as rc
from skl2onnx.common.data_types import FloatTensorType

# -------------------------------------------------------------------------------------------

# settings

# clustercollection = 'CaloClusters'
trainingTool = 'lgbm'  # xgboost, lgbm
# trainingTool = 'xgboost'  # xgboost, lgbm
useROOT = False
debug = False
scaleInputs = False  # did not find any improvement in AUC with rescaling, and complicates applying trained BDT after some Ecl cuts
do1DPlots = False
doTraining = True

# default training settings
learning_rate = 0.005
num_leaves = 63
max_bin = 255
# max_depth = -1  # note that this will lead to massive overfitting with xgboost
max_depth = 6
if max_depth == -1 and trainingTool == 'xgboost':
    max_depth = 0
# max_bin = 63
# max_depth = 4
# min_data_in_bin = 1    # or min_data_in_leaf?
early_stopping_rounds = 30
epochs = 10000
feature_fraction = 0.5
# feature_fraction = 0.9
metric_frequency = 10 # print auc every 10 events
# a json file that might override the training settings
trainingParams = ''

# the folder with the pi0 and gamma root files, the file and tree names
# basedir = "/home/lit/public/allegro/photonID_sample/strip_L3_1M/"
# basedir = "/home/lit/public/giovanni/sample/photonID_stripL3/production_reconstruction/"
# basedir = os.getcwd() + "/"
basedir = "/home/gmarchio/work/fcc/allegro/fullsim/run/test/training_reconstruction_3/"
gamma_filename = "production_reconstruction_particle_gamma.root"
pi0_filename = "production_reconstruction_particle_pi0.root"

treename = 'events'
shapeParameterNames = []  # to be readout from the files

# the shape variables that should not be used
# cluster barycenters
columns_to_drop1 = ['theta_EMB_layer_{:d}'.format(i) for i in range(0, 12)]
columns_to_drop2 = ['phi_EMB_layer_{:d}'.format(i) for i in range(0, 12)]
columns_to_drop3 = ['dR_over_E']  # for topoclusters
columns_to_drop = ['icl'] + columns_to_drop1 + columns_to_drop2 + columns_to_drop3


# -------------------------------------------------------------------------------------------

if trainingTool == 'xgboost':
    import xgboost as xgb
elif trainingTool == 'lgbm':
    import lightgbm as lgb
else:
    print('Unknown training tool', trainingTool)
    exit(0)


# -------------------------------------------------------------------------------------------


# read and return list of shower shape parameters from file
def read_metadata(filename, clusters):
    # print('Reading metadata in file', filename)
    shapeParameterNames = []

    # Invoke the shell script and capture the output
    output = subprocess.check_output(['./getMetaData.sh', filename, clusters], shell=False)

    # Decode the output (it's a bytes object) and split it into lines
    lines = output.decode().splitlines()

    for line in lines:
        shapeParameterNames.append(line.split()[1])
    return shapeParameterNames

# -------------------------------------------------------------------------------------------


# read file and return pandas dataframe containing cluster energy and shower shape parameters
# this thing could be more efficient using RDataFrame, I suspect..
# should check code by Nicolas
def read_events(filename, treename, emin, emax, bdtclass, **kwargs):

    column_names = ['icl', 'ecl']
    if 'shapeParamsIndex' in kwargs:
        # if list of indices of shapeParams is passed, read only those variables
        shapeParamsIndex = kwargs['shapeParamsIndex']
        column_names.extend([shapeParameterNames[i] for i in shapeParamsIndex])
    elif 'shapeParamsName' in kwargs:
        # if list of names of shapeParams is passed, read only those variables
        shapeParamsName = kwargs['shapeParamsName']
        shapeParamsIndex = []
        for name in shapeParamsName:
            try:
                index = shapeParameterNames.index(name)
                shapeParamsIndex.append(index)
            except ValueError:
                print(f'{name} not found in the list')
        column_names.extend(shapeParamsName)
    else:
        column_names.extend(shapeParameterNames)
        shapeParamsIndex = [i for i in range(0, len(shapeParameterNames))]

    print('\nNames of the columns of the dataframe to be created:')
    print(column_names)
    print('\nIndex of the shape parameters to be read, based on metadata information:')
    print(shapeParamsIndex)

    clusters = "Augmented"+clustercollection
    if useROOT:
        import ROOT
        print('\nProcessing file:', filename)
        print('Reading tree:', treename)
        df = ROOT.ROOT.RDataFrame("events", filename)
        num_init = df.Count()
        df = df.Alias("clusters_energy", f"{clusters}.energy")
        df = (
            df
            .Define("icl", "ArgMax(clusters_energy)")
            .Define("ecl", "clusters_energy[icl]")
            .Define("shapeParameters_begin_cl", f"{clusters}.shapeParameters_begin[icl]")
        )
        for iPar in shapeParamsIndex:
            df = df.Define(shapeParameterNames[iPar], f"_{clusters}_shapeParameters[shapeParameters_begin_cl + {iPar}]")

        d = df.Filter(f"icl>=0 && ecl>{emin} && ecl<{emax}")
        cols = d.AsNumpy(column_names)
        print("We have run on", num_init.GetValue(), "events")
        df = pd.DataFrame(cols)
    else:
        # open file
        print('\nProcessing file:', filename)
        afile = uproot.open(filename)

        # read tree
        print('Reading tree:', treename)
        tree = afile[treename]

        # read branches
        arr = tree.arrays(
            [
                f'{clusters}/{clusters}.energy',
                f'{clusters}/{clusters}.shapeParameters_begin',
                f'_{clusters}_shapeParameters',
            ],
            library='np')
        ecl = arr[f'{clusters}/{clusters}.energy']
        parBegin = arr[f'{clusters}/{clusters}.shapeParameters_begin']
        shapePars = arr[f'_{clusters}_shapeParameters']

        # loop over tfile events and fill dataframe
        nentries = len(ecl)
        print('Number of entries:', nentries)
        data = []
        skipped_entries = []
        for entry in tqdm(range(nentries), mininterval=1):
            if len(ecl[entry]) == 0:
                # print('No clusters found for event %d, skipping' % entry)
                skipped_entries.append(entry)
            else:
                icl = np.argmax(ecl[entry])
                e = ecl[entry][icl]
                if e < emin or e > emax:
                    continue
                new_row = [icl, ecl[entry][icl]]
                index_first_feature = parBegin[entry][icl]
                features = shapePars[entry]
                if debug:
                    print('Cluster index:', icl)
                    print('Cluster energy:', ecl[entry][icl])
                    print('Cluster shapeParameters_begin:', parBegin[entry][icl])

                for iPar in shapeParamsIndex:
                    index_feature = index_first_feature + iPar
                    feature = features[index_feature]
                    new_row.append(feature)
                data.append(new_row)
        if len(skipped_entries) > 0:
            print('No clusters found for events', skipped_entries)

        # create pandas dataframe from data (list of lists)
        df = pd.DataFrame(data, columns=column_names)

    df['class'] = bdtclass
    print('Dataframe content:')
    print(df)
    return df

# -------------------------------------------------------------------------------------------


# plot feature importance of variables in BDT
def plotImp(model, X, num=20, fig_size=(40, 20), imptype="gain"):
    outfile = f'{outdir}/plots/bdt-photonid-importances-{clustercollection}.pdf'
    print("\nPlotting the ranked feature importance to %s ...\n" % outfile)
    # If split, result contains numbers of times the feature is used in a model.
    # If gain, result contains total gains of splits which use the feature
    if trainingTool == 'lgbm':
        feature_imp = pd.DataFrame({'Value': model.feature_importance(importance_type=imptype), 'Feature': X.columns})
    elif trainingTool == 'xgboost':
        feature_important = model.get_score(importance_type=imptype)  # could also be weight
        values = list(feature_important.values())
        feature_imp = pd.DataFrame({'Value': values, 'Feature': X.columns})

    fig = plt.figure()
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Feature importance (%s)' % imptype)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)
    # save full list of feature importance to file
    feature_imp.sort_values(by='Value', ascending=False).to_csv(f'{outdir}/models/bdt-photonid-importances-{clustercollection}.csv', index=False, sep=" ")

# -------------------------------------------------------------------------------------------

# helper functions for 1D plots of input variables


def title(var):
    if var == 'ecl':
        return '$E_{cl}$ [GeV]'
    elif var == 'icl':
        return 'Cluster index'
    elif var == 'mass':
        return '$m_{cl}$ [GeV]'
    elif var == 'ncells':
        return '$N_{cells}$'
    elif var == 'class':
        return var
    else:
        layer = int(var.split('_')[-1])
        if var.startswith('theta_EMB_layer'):
            return '$\\theta ({L%d})$' % layer
        elif var.startswith('phi_EMB_layer'):
            return '$\\phi ({L%d})$' % layer
        elif var.startswith('maxcell_E_EMB'):
            return '$E^{max cell} ({L%d})$' % layer
        elif var.startswith('2ndmaxcell_E_EMB'):
            return '$E^{2nd max cell} ({L%d})$' % layer
        elif var.startswith('width_theta_EMB_layer'):
            return '$w_{\\theta} ({L%d})$' % layer
        elif var.startswith('width_module_EMB_layer'):
            return '$w_{\\phi} ({L%d})$' % layer
        elif var.startswith('width_theta_3Bin_EMB_layer'):
            return '$w_{\\theta}^{3} ({L%d})$' % layer
        elif var.startswith('width_theta_5Bin_EMB_layer'):
            return '$w_{\\theta}^{5} ({L%d})$' % layer
        elif var.startswith('width_theta_7Bin_EMB_layer'):
            return '$w_{\\theta}^{7} ({L%d})$' % layer
        elif var.startswith('width_theta_9Bin_EMB_layer'):
            return '$w_{\\theta}^{9} ({L%d})$' % layer
        elif var.startswith('Delta_E_2ndmax_min_EMB_layer'):
            return '$\\Delta E_{\\theta} ({L%d})$' % layer
        elif var.startswith('Delta_E_2ndmax_min_vs_phi_EMB_layer'):
            return '$\\Delta E_{\\phi} ({L%d})$' % layer
        elif var.startswith('energy_fraction_EMB_layer'):
            return '$E(L%d)/E_{cl}$' % layer
        elif var.startswith('Ratio_E_max_2ndmax_EMB_layer'):
            return '$E_{ratio}^{\\theta}$ (L%d)' % layer
        elif var.startswith('Ratio_E_max_2ndmax_vs_phi_EMB_layer'):
            return '$E_{ratio}^{\\phi}$ (L%d)' % layer
        elif var.startswith('E_fr_side_pm2_EMB_layer'):
            return '$f_{side}^{2}$ (L%d)' % layer
        elif var.startswith('E_fr_side_pm3_EMB_layer'):
            return '$f_{side}^{3}$ (L%d)' % layer
        elif var.startswith('E_fr_side_pm4_EMB_layer'):
            return '$f_{side}^{4}$ (L%d)' % layer
        else:
            return var


def plotrange(var):
    if var == 'ecl':
        return (0, 100)
    elif var == 'ncells':
        return (0, 1000)
    elif var == 'mass':
        return (0, 2.0)
    elif var.startswith('theta_EMB_layer'):
        return (0, 3.1416)
    elif var.startswith('phi_EMB_layer'):
        return (-3.1416, 3.1416)
    elif var.startswith('maxcell_E_EMB'):
        return (0, 10)
    elif var.startswith('2ndmaxcell_E_EMB'):
        return (0, 0.1)
    elif var.startswith('width_theta_EMB_layer'):
        return (0, 20)
    elif var.startswith('width_module_EMB_layer'):
        return (0, 20)
    elif var.startswith('width_theta_3Bin_EMB_layer'):
        return (0, 4)
    elif var.startswith('width_theta_5Bin_EMB_layer'):
        return (0, 5)
    elif var.startswith('width_theta_7Bin_EMB_layer'):
        return (0, 6)
    elif var.startswith('width_theta_9Bin_EMB_layer'):
        return (0, 7)
    elif var.startswith('Delta_E_2ndmax_min_EMB_layer'):
        return (0, 0.5)
    elif var.startswith('Delta_E_2ndmax_min_vs_phi_EMB_layer'):
        return (0, 2.0)
    elif var.startswith('energy_fraction_EMB_layer'):
        return (0, 0.5)
    else:
        return (0, 1)


# make comparison plots of the shower shapes
def plot_vars(gamma_df, pi0_df):

    # Create a PdfPages object to save multi-page PDF
    varlist = list(gamma_df.keys())
    nvars = len(varlist)
    ncols = 4
    nrows = 3
    varsperpage = ncols * nrows
    npages = math.ceil(nvars / varsperpage)
    print('Number of variables to plot:', nvars)
    print('Variables per page: %d (%d x %d)' % (varsperpage, nrows, ncols))
    print('Number of pages to print:', npages)
    with PdfPages('{outdir}/plots/showershapes-{clustercollection}.pdf') as pdf:
        for ipage in range(npages):
            print('Page: ', ipage)
            if ipage < npages - 1:
                varstoplot = varsperpage
            else:
                varstoplot = ((nvars - 1) % varsperpage) + 1
            # Create the page with the subplots
            fig, axs = plt.subplots(nrows, ncols, figsize=(16, 12))
            for i, ax in enumerate(axs.flat):
                if i < varstoplot:
                    ivar = varsperpage * ipage + i
                    varname = varlist[ivar]
                    vartitle = title(varname)
                    ax.hist(gamma_df[varname], range=plotrange(varname), bins=50, alpha=0.5, label='$\\gamma$')
                    ax.hist(pi0_df[varname], range=plotrange(varname), bins=50, alpha=0.5, label='$\\pi^{0}$')
                    ax.set_xlabel(vartitle)
                    # Add a legend
                    ax.legend(loc='upper right')
                    # ax.set_title(vartitle)

            # Save the page of the PDF
            pdf.savefig(fig)

            # Save each of the first set of subplots individually
            for i, ax in enumerate(axs.flat):
                if i < varstoplot:
                    ivar = varsperpage * ipage + i
                    varname = varlist[ivar]
                    vartitle = title(varname)
                    fig_single, ax_single = plt.subplots()
                    ax_single.hist(gamma_df[varname], range=plotrange(varname), bins=50, alpha=0.5, label='$\\gamma$')
                    ax_single.hist(pi0_df[varname], range=plotrange(varname), bins=50, alpha=0.5, label='$\\pi^{0}$')
                    ax_single.set_xlabel(vartitle)
                    ax_single.legend(loc='upper right')
                    fig_single.savefig(f'{outdir}/plots/{varname}-{clustercollection}.pdf')
                    plt.close(fig_single)

            plt.close(fig)


# -------------------------------------------------------------------------------------------


# Find the rows and columns of all missing values
def find_nan(df):
    # Reshape the DataFrame into a Series
    # series = df.stack(dropna=False)  # this will keep NaN - useful for debugging, but will raise a warning
    series = df.stack(future_stack=True)  # this won't keep rows with NaN
    # Select the NaN values and their corresponding row and column labels
    missing_values = series[series.isna()]
    # Print the row and column labels of the NaN values
    print('List of elements in dataframe which are NaN (if any):')
    print(missing_values.index)


# -------------------------------------------------------------------------------------------


# remove columns not used for training
# check for NaNs
def clean_dfs(gamma_df, pi0_df):
    # drop all columns for Ratio_E_max_2ndmax_vs_phi: a lot of NaN
    # sanity check: drop theta and phi of cluster barycenter if they were included in initial list of features, as well as cluster index
    columns_to_drop1 = gamma_df.columns[gamma_df.columns.str.startswith('theta_EMB_layer')]
    columns_to_drop2 = gamma_df.columns[gamma_df.columns.str.startswith('phi_EMB_layer')]
    columns_to_drop = ['icl'] + columns_to_drop1.tolist() + columns_to_drop2.tolist() + ['dR_over_E']
    gamma_df = gamma_df.drop(columns_to_drop, axis=1)
    pi0_df = pi0_df.drop(columns_to_drop, axis=1)

    # can also do
    # gamma_df = gamma_df.dropna()
    # pi0_df = pi0_df.dropna()

    # or replace with fixed value
    # gamma_df.fillna(-99, inplace=True)
    # pi0_df.fillna(-99, inplace=True)

    # check that there are no further NaN
    find_nan(gamma_df)
    find_nan(pi0_df)

    return gamma_df, pi0_df


# -------------------------------------------------------------------------------------------

# train and return a model given a dataframe

def trainBDT(df):

    # print the input dataset
    print(df)

    # define target variable (y) and input features (X)
    y = df['class']
    X = df.drop('class', axis=1)

    if trainingTool == 'xgboost':
        # rename features or otherwise export to onnx will fail
        new_column_names = {old_name: f'f{i}' for i, old_name in enumerate(X.columns)}
        X = X.rename(columns=new_column_names)
        print(X)

    # split the input dataset in train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # scale input variables
    if scaleInputs:
        print('Scaling the inputs')
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Save the arrays to a .npz file
    np.savez(f'{outdir}/models/bdt-photonid-inputs-train-test-split-{clustercollection}.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, columns=X.columns)

    # loading data
    print(f"\nLoading the test and train datasets into {trainingTool} classifier...")
    params = {}
    if trainingTool == 'lgbm':
        # No weights
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
        # lgb_train = lgb.Dataset(x_train, y_train, weight=w_train)
        # lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train, weight=w_test)

        # defining parameters
        if trainingParams != '':
            file = open(trainingParams, 'r')
            params = json.loads(file.read())
            file.close()
            os.cmd(f'cp -f {trainingParams} {outdir}/models/bdt-photonid-trainingparams-{clustercollection}.json')
        else:
            params = {
                # 'task': 'train',
                'objective': 'binary',
                'boosting': 'gbdt',
                'metric': ['auc'],
                'learning_rate': learning_rate,
                'num_leaves': num_leaves,
                'max_bin': max_bin,
                # 'use_quantized_grad': True,
                'metric_freq': metric_frequency,
                'early_stopping_rounds': early_stopping_rounds,
                'num_iteration': epochs,
                'max_depth': max_depth,
                # 'min_data_in_bin': min_data_in_bin,  # or min_data_in_leaf? # should be probably set to avoid warnings about best gain: -inf
                'feature_fraction': feature_fraction,
                'is_provide_training_metric': True,
                # 'num_threads': 24,
                'verbosity': 0,
                'device': 'gpu'
            }
    elif trainingTool == 'xgboost':
        initial_model = xgb.XGBClassifier(n_estimators=epochs,
                                          max_depth=max_depth,
                                          learning_rate=learning_rate,
                                          subsample=0.7,
                                          colsample_bytree=0.7,
                                          colsample_bylevel=0.7,
                                          base_score=y_train.mean(),
                                          tree_method="hist",
                                          device="cuda",
                                          # eval_metric='error',
                                          eval_metric='auc',
                                          objective="binary:logistic",
                                          early_stopping_rounds=early_stopping_rounds,
                                          random_state=42, seed=42)
        params = initial_model.get_params()

    # Write to JSON file
    outfile = f'{outdir}/models/bdt-photonid-trainingparams-{clustercollection}.json'
    file = open(outfile, 'w')
    json.dump(params, file) # , default=list) - only needed if some param is a set
    file.close()

    dataForJson["trainingParameters"] = params
    with open(f'{outdir}/models/bdt-photonid-settings-{clustercollection}.json', 'w') as json_file:
        json.dump(dataForJson, json_file, indent=4)

    print('\nTraining parameters:')
    print(params)


    # fit the model and plot the history of the training
    print("\nPerforming the training...")
    outfile = f'{outdir}/plots/bdt-photonid-training-history-auc-{clustercollection}.pdf'
    print("\nThe training history will be recorded in file %s" % outfile)
    model = None
    if trainingTool == 'lgbm':
        evals = {}
        model = lgb.train(params,
                          train_set=lgb_train,
                          valid_sets=[lgb_train, lgb_test],
                          valid_names=["train", "test"],
                          callbacks=[lgb.log_evaluation(metric_frequency), lgb.record_evaluation(evals)],
                          )

        # Plot the history of the training
        lgb.plot_metric(evals, metric='auc')
        plt.savefig(outfile)
        # plt.show()
    elif trainingTool == 'xgboost':
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model = initial_model.fit(X_train, y_train,
                                  eval_set=eval_set,
                                  verbose=10)

        # Extract the AUC history from the evaluation results
        train_auc = model.evals_result()['validation_0']['auc']
        test_auc = model.evals_result()['validation_1']['auc']

        # Plot the history of the training
        plt.plot(range(1, len(train_auc) + 1), train_auc, label='Train AUC')
        plt.plot(range(1, len(test_auc) + 1), test_auc, label='Test AUC')
        # plt.ylim([0.0, 1.0])
        plt.xlabel('Number of Boosting Rounds')
        plt.ylabel('AUC')
        plt.title('AUC History')
        plt.legend()
        plt.savefig(outfile)
        # plt.show()

    # save model
    extension = 'txt' if trainingTool == 'lgbm' else 'json'
    outfile = f'{outdir}/models/bdt-photonid-weights-{clustercollection}.{extension}'
    print("\nSaving the model to file %s ..." % outfile)
    model.save_model(outfile)

    outfile = f'{outdir}/models/bdt-photonid-weights-{clustercollection}.onnx'
    print("\nSaving the model to file %s ..." % outfile)
    if trainingTool == 'lgbm':
        model_onnx = onnxmltools.convert_lightgbm(model,
                                                  initial_types=[('X', FloatTensorType([None, X_train.shape[1]]))],
                                                  zipmap=False,
                                                  split=100)
    elif trainingTool == 'xgboost':
        model_onnx = onnxmltools.convert_xgboost(model,
                                                 initial_types=[('X', FloatTensorType([None, X_train.shape[1]]))])

    onnx.save(model_onnx, outfile)
    # note: the saved model has a single input tensor called "X", with size equal to the number of input features
    # that's why elsewhere in this script we need to save the name of the input features as a separate file
    # with XGBoost instead one could save the model as using N tensors of size 1 as input, each one with its own name,
    # with
    # input_names = X.columns.tolist()
    # model_onnx = onnxmltools.convert_xgboost(model,
    #                                          initial_types=[('float_input', FloatTensorType([None, 1]))] * len(input_names),
    #                                          input_names=input_names,
    #                                          target='class')



    return model, X_train, X_test, y_train, y_test

# -------------------------------------------------------------------------------------------

# Define a custom evaluation function (NOT USED)

def eval_auc(preds, dtrain):
    labels = dtrain.get_label()
    preds = preds.reshape(-1, 1)
    auc = ras(labels, preds)
    return 'auc', auc

# -------------------------------------------------------------------------------------------

def get_bkg_eff(sig_eff, tpr, fpr):
    # Find the index of the tpr array that is closest to sig_eff
    idx = np.argmin(np.abs(tpr - sig_eff))

    # If the TPR array does not contain sig_eff exactly, interpolate the FPR array
    # to find the FPR value at TPR = sig_eff
    if tpr[idx] != sig_eff:
        bkg_eff = np.interp(sig_eff, tpr[idx:idx + 2], fpr[idx:idx + 2])
    else:
        bkg_eff = fpr[idx]

    # Print the FPR value at given TPR
    print('pi0 efficiency at photon efficiency = {:.1f}%: {:.1f}%'.format(100 * sig_eff, 100 * bkg_eff))

    return bkg_eff

# -------------------------------------------------------------------------------------------


# main program

parser = argparse.ArgumentParser(description='Optimise BDT for photon/pi0 discrimination')
parser.add_argument('--emin', type=int, default=0, help='The minimum cluster energy')
parser.add_argument('--emax', type=int, default=1000, help='The minimum cluster energy')
parser.add_argument('--outdir', type=str, default='inclusive', help='The output folder (optional)')
parser.add_argument('--skipTraining', action='store_true', help='pass this option to skip the training')
parser.add_argument('--overwriteDir', action='store_true', help='pass this option to overwrite an existing output directory during the training')
parser.add_argument('--indir', type=str, default='', help='The folder where the BDT to test is located (this option is only valid when skipTraining is enabled)')
parser.add_argument('--trainingParams', type=str, default='', help='An optional json file containing the parameters for the BDT training')
parser.add_argument('--clusters', type=str, default='CaloClusters', help='The input cluster collection')

args = parser.parse_args()
emin = int(args.emin)
emax = int(args.emax)
outdir = args.outdir
overwriteDir = args.overwriteDir
# print(args.skipTraining)
doTraining = not args.skipTraining
indir = args.indir
trainingParams = args.trainingParams
clustercollection = args.clusters

if doTraining:

    if indir != '':
        print('Configuration error: options indir and skipTraining cannot be set at the same time')
        exit(0)
    indir = outdir

    if os.path.isdir(outdir):
        print('Output directory already exists, if you continue the previous training will be overwritten')
        if not overwriteDir:
            print('If you really want to overwrite the previous training, delete the directory or pass the --overwriteDir flag')
            exit(0)
    else:
        os.mkdir(outdir)
        os.mkdir(outdir + '/plots')
        os.mkdir(outdir + '/models')

    dataForJson = {
        "timeStamp": str(datetime.datetime.now()),
        "clusterCollection": clustercollection,
        "trainingTool": trainingTool,
    }

    # read the metadata in the gamma and pi0 files and make sure they are the sam
    shapeParameterNames_gamma = read_metadata(basedir + gamma_filename, f'Augmented{clustercollection}')
    shapeParameterNames_pi0 = read_metadata(basedir + pi0_filename,  f'Augmented{clustercollection}')
    assert shapeParameterNames_gamma == shapeParameterNames_pi0, 'shower shape decorations must be the same in the two files'
    shapeParameterNames = shapeParameterNames_gamma
    # write the list of parameter names used at training time to the models directory
    outfile = f'{outdir}/models/metadata-{clustercollection}.json'
    print('\nMetadata have been read correctly from the two files. Writing to output file', outfile)
    file = open(outfile, 'w')
    json.dump(shapeParameterNames, file)
    file.close()

    trainingVars = ['ecl']
    trainingVars.extend([x for x in shapeParameterNames if x not in columns_to_drop])
    outfile = f'{outdir}/models/bdt-photonid-inputs-{clustercollection}.json'
    file = open(outfile, 'w')
    json.dump(trainingVars, file)
    file.close()

    # read the photon and pi0 files into pandas dataframes
    # add the column with the class (0 for bkg, 1 for signal) for the BDT
    # print the content of the datasets
    gamma_df = read_events(basedir + gamma_filename, treename, emin, emax, 1)
    pi0_df = read_events(basedir + pi0_filename, treename, emin, emax, 0)

    # example of how to read only some variables:
    # gamma_df = read_events(gamma_filename, treename, shapeParamsIndex=[0,1,2,3])
    # pi0_df = read_events(pi0_filename, treename, shapeParamsName=['maxcell_E_EMB_layer_1', 'Delta_E_2ndmax_min_vs_phi_EMB_layer_8', 'width_theta_EMB_layer_10'])
    # or:
    # varsToUse = {
    #     'Delta_E_2ndmax_min_EMB_layer_3',
    #     'width_theta_5Bin_EMB_layer_3',
    #     'mass',
    # } - {'ecl'}
    # gamma_df = read_events(basedir + gamma_filename, treename, emin, emax, 1, shapeParamsName=varsToUse)
    # pi0_df = read_events(basedir + pi0_filename, treename, emin, emax, 0, shapeParamsName=varsToUse)

    # plot the vars
    if do1DPlots:
        print('Creating 1D plots of the input variables')
        plot_vars(gamma_df, pi0_df)

    # check if there are NaN
    find_nan(gamma_df)
    find_nan(pi0_df)

    # drop all columns for Ratio_E_max_2ndmax_vs_phi: a lot of NaN
    # sanity check: drop theta and phi of cluster barycenter if they were included in initial list of features, as well as cluster index
    gamma_df, pi0_df = clean_dfs(gamma_df, pi0_df)

    # create the combined dataset
    df = pd.concat([gamma_df, pi0_df], ignore_index=True)
    column_names = df.columns[:-1]
    featureNames = column_names.copy()
    dataForJson["shapeParameters"] = list(column_names)

    # do the training
    model, X_train, X_test, y_train, y_test = trainBDT(df)

else:
    # Read the previously trained bdt
    if indir == '':
        indir = outdir
    if trainingTool == 'lgbm':
        model = lgb.Booster(model_file=f'{indir}/models/bdt-photonid-weights-{clustercollection}.txt')
    elif trainingTool == 'xgboost':
        model = xgb.Booster(model_file=f'{indir}/models/bdt-photonid-weighs-{clustercollection}.json')
    # Read the data
    if os.path.isfile(f'{indir}/models/bdt-photonid-inputs-train-test-split-{clustercollection}.npz'):
        # Load the arrays from the .npz file
        data = np.load(f'{indir}/models/bdt-photonid-inputs-train-test-split-{clustercollection}.npz', allow_pickle=True)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        column_names = data['columns']
        # apply extra cut on emin<e<emax
        condition1 = X_test[:, 0] < emax
        condition2 = X_test[:, 0] > emin
        condition = np.logical_and(condition1, condition2)
        X_test = X_test[condition]
        y_test = y_test[condition]
        print('Size of train dataset:', len(y_train))
        print('Size of test dataset:', len(y_test))
        print('')
        # debug
        # print(condition1)
        # print(condition2)
        # print(condition)
        # print(X_test)
        # print(y_test)

    else:
        # Read the data from scratch
        # maybe not a very good idea because some events in the test sample could actually have been used to train the BDT?

        # read the list of parameter names used at training time from the models directory
        # file = open(outdir + '/models/shapeParameters.json', 'r')
        # shapeParameterNames = json.loads(file.read())
        # file.close()
        # check that it matches that of the file that we want to use for estimating the BDT performance
        # shapeParameterNames_gamma = read_metadata(basedir + gamma_filename)
        # assert shapeParameterNames_gamma == shapeParameterNames, 'shower shape decorations must be the same in the training and test samples'
        # read the trees
        # gamma_df = read_events(basedir + gamma_filename, treename, emin, emax, 1)
        # pi0_df = read_events(basedir + pi0_filename, treename, emin, emax, 0)
        # clean_dfs(gamma_df, pi0_df)
        # y = df['class']
        # X = df.drop('class', axis=1)
        # split the input dataset in train and test samples
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        pass

X_concatenated = np.concatenate((X_train, X_test), axis=0)
X = pd.DataFrame(X_concatenated, columns=column_names)

# calculate probabilities, for AUC
# for xgboost, reread model from file to run in cpu
if trainingTool == 'xgboost':
    model = xgb.Booster(model_file=f'{indir}/models/bdt-photonid-weights-{clustercollection}.json')
    X_train = xgb.DMatrix(X_train)
    X_test = xgb.DMatrix(X_test)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
test_auc = ras(y_test, y_test_pred)
train_auc = ras(y_train, y_train_pred)
print("Training ROC-AUC:", train_auc)
print("Test ROC-AUC:", test_auc)

# plot ROC curve
fpr, tpr, _ = rc(y_test, y_test_pred)

# print(fpr)
# print(tpr)

bdt_score_train_gamma = [element for element, flag in zip(y_train_pred, y_train) if flag == 1]
bdt_score_train_pi0 = [element for element, flag in zip(y_train_pred, y_train) if flag == 0]
bdt_score_test_gamma = [element for element, flag in zip(y_test_pred, y_test) if flag == 1]
bdt_score_test_pi0 = [element for element, flag in zip(y_test_pred, y_test) if flag == 0]

# plot BDT score for signal and background, test and train
fig_single, ax_single = plt.subplots()
ax_single.hist(bdt_score_test_gamma, range=(0, 1), bins=50, alpha=0.5, label='$\\gamma$ (test)', density=True)
ax_single.hist(bdt_score_test_pi0, range=(0, 1), bins=50, alpha=0.5, label='$\\pi^{0}$ (test)', density=True)
ax_single.hist(bdt_score_train_gamma, range=(0, 1), bins=50, alpha=0.5, label='$\\gamma$ (train)', density=True, histtype='step', edgecolor='blue', linewidth=2)
ax_single.hist(bdt_score_train_pi0, range=(0, 1), bins=50, alpha=0.5, label='$\\pi^{0}$ (train)', density=True, histtype='step', edgecolor='red', linewidth=2)
ax_single.set_xlabel('BDT probability')
ax_single.legend(loc='upper right')
outfile = f'{outdir}/plots/bdt-photonid-score-lin-{clustercollection}.pdf'
print("\nDrawing the BDT score distribution to %s ..." % outfile)
fig_single.savefig(outfile)
ax_single.semilogy()
outfile = f'{outdir}/plots/bdt-photonid-score-log-{clustercollection}.pdf'
print("\nDrawing the BDT score distribution to %s ..." % outfile)
fig_single.savefig(f'{outdir}/plots/bdt-photonid-score-log-{clustercollection}.pdf')
plt.close(fig_single)

# Compute 1 - FPR
one_minus_fpr = 1 - fpr
# Plot  ROC curve with TPR on x-axis (eff_gamma) and 1 - FPR (1-eff_pi0) on y-axis
fig = plt.figure()
plt.plot(tpr, one_minus_fpr, color='blue', lw=2, label='ROC curve (AUC=%.3f)' % test_auc)
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
outfile = f'{outdir}/plots/bdt-photonid-roc-{clustercollection}.pdf'
print("\nDrawing the ROC curve to %s ..." % outfile)
plt.savefig(outfile)
plt.close(fig)

# round probabilities, for accuracy
y_train_pred = y_train_pred.round()
y_test_pred = y_test_pred.round()
print("Accuracy (train)", accuracy_score(y_train, y_train_pred))
print("Accuracy (test)", accuracy_score(y_test, y_test_pred))

# calculate and plot confusion matrix
outfile = f'{outdir}/plots/bdt-photonid-confusion-validation-{clustercollection}.pdf'
print("\nDrawing the confusion matrix on the validation sample to %s ..." % outfile)
cm = confusion_matrix(y_test, y_test_pred)
#                      labels=np.arange(len(channels)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[r"$\pi^{0}$", r"$\gamma$"])
disp.plot()
# plt.show()
disp.figure_.savefig(outfile)
disp.figure_.clear()

outfile = f'{outdir}/plots/bdt-photonid-confusion-norm-validation-{clustercollection}.pdf'
print("\nDrawing the normalised confusion matrix on the validation sample to %s ..." % outfile)
cm2 = 100 * cm / cm.astype(float).sum(axis=1)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                               display_labels=[r"$\pi^{0}$", r"$\gamma$"])
disp2.plot(xticks_rotation="vertical", values_format=".1f")
# plt.show()
disp2.figure_.savefig(outfile)
disp2.figure_.clear()

# plot the feature importance
plotImp(model, pd.DataFrame(X))

# print bkg eff at some benchmark sig eff
get_bkg_eff(0.8, tpr, fpr)
get_bkg_eff(0.9, tpr, fpr)
get_bkg_eff(0.95, tpr, fpr)
get_bkg_eff(0.99, tpr, fpr)
print("\n")

# a sanity check: calculate score for given set of values
# X_test = [[
#     5.48329,
#     0,
#     0,
#     0,
#     0,
#     1,
#     0,
#     1,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0.00104011,
#     0.00328055,
#     3.59048,
#     2.17338,
#     0.497176,
#     0.00110177,
#     0.233652,
#     0.00201852,
#     0,
#     0.152744,
#     0.152744,
#     0.152744,
#     0.00590169,
#     0.00590169,
#     0.00590169,
#     0.0857333,
#     0.144496,
#     2.19813,
#     1.11034,
#     1,
#     0,
#     1,
#     0,
#     2.11962,
#     2.19813,
#     2.19813,
#     2.19813,
#     0.00931629,
#     0.00931629,
#     0.00931629,
#     0.263539,
#     0.359661,
#     2.41452,
#     2.11007,
#     1,
#     0,
#     0.953805,
#     0.0153048,
#     2.21611,
#     2.37632,
#     2.41452,
#     2.41452,
#     0.00904036,
#     0.0100852,
#     0.0100852,
#     0.146291,
#     0.197047,
#     3.06487,
#     2.03066,
#     0.974761,
#     0.00501531,
#     0.965742,
#     0.00427359,
#     2.33121,
#     2.57401,
#     2.71469,
#     3.06487,
#     0.0295943,
#     0.037381,
#     0.0444967,
#     0.18449,
#     0.311247,
#     4.1398,
#     2.16057,
#     0.836831,
#     0.00638236,
#     0.979624,
#     0.00426831,
#     2.50763,
#     3.11451,
#     4.06441,
#     4.1398,
#     0.0668711,
#     0.130757,
#     0.134239,
#     0.14921,
#     0.233848,
#     3.78025,
#     2.04659,
#     0.916871,
#     0.0210269,
#     0.855092,
#     0.00865665,
#     2.2001,
#     2.82066,
#     2.87179,
#     3.78025,
#     0.0565055,
#     0.0588315,
#     0.088291,
#     0.0975056,
#     0.112182,
#     4.20061,
#     3.02834,
#     0.919277,
#     0.00301903,
#     0.415494,
#     0.00569775,
#     2.88811,
#     3.50707,
#     3.75774,
#     4.20061,
#     0.0678095,
#     0.0836306,
#     0.101543,
#     0.0338772,
#     0.0592747,
#     5.04143,
#     2.08081,
#     1,
#     0,
#     0.999625,
#     1.71178e-05,
#     2.4036,
#     4.35678,
#     4.77759,
#     5.04143,
#     0.326287,
#     0.373671,
#     0.392666,
#     0.0258005,
#     0.0290386,
#     5.93032,
#     5.34423,
#     0.716028,
#     0.00957206,
#     0.385026,
#     0.017644,
#     2.2089,
#     3.40122,
#     4.1891,
#     4.24513,
#     0.245995,
#     0.357839,
#     0.362265,
#     0.0125132,
#     0.0288637,
#     7.29503,
#     7.05367,
#     0.388192,
#     0.00452264,
#     0.554582,
#     0.00827002,
#     1.80265,
#     3.47514,
#     3.79687,
#     7.29503,
#     0.332397,
#     0.348239,
#     0.566541,
#     0.0748231,
#     208
# ]]
# y_pred = model.predict(X_test)
# print(y_pred)
