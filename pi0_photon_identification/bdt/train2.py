# -------------------------------------------------------------------------------------------
#
# train.py
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
from sklearn.model_selection import RandomizedSearchCV

# -------------------------------------------------------------------------------------------

# settings

# trainingTool = 'lgbm'  # xgboost, lgbm
trainingTool = 'xgboost'  # xgboost, lgbm
useROOT = True
debug = False
scaleInputs = True  # did not find any improvement in AUC with rescaling with LGBM, and complicates applying trained BDT after some Ecl cuts
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
epochs = 1000
feature_fraction = 0.5
# feature_fraction = 1.0
# a json file that might override the training settings
trainingParams = ''

# the folder with the pi0 and gamma root files, the file and tree names
basedir = "/home/lit/public/allegro/photonID_sample/strip_L3_1M/"
# basedir = "/home/lit/public/giovanni/sample/photonID_stripL3/production_reconstruction/"
# basedir = os.getcwd() + "/"
gamma_filename = "production_reconstruction_particle_gamma_1M.root"
pi0_filename = "production_reconstruction_particle_pi0_1M.root"
treename = 'events'
shapeParameterNames = []  # to be readout from the files

# the shape variables that should not be used
# cluster barycenters
columns_to_drop1 = ['theta_EMB_layer_{:d}'.format(i) for i in range(0, 12)]
columns_to_drop2 = ['phi_EMB_layer_{:d}'.format(i) for i in range(0, 12)]
# Eratio vs phi (NaN due to bug, to be fixed in new prod)
# columns_to_drop3 = ['Ratio_E_max_2ndmax_vs_phi_EMB_layer_{:d}'.format(i) for i in range(0,11)]
columns_to_drop = ['icl'] + columns_to_drop1 + columns_to_drop2
# + columns_to_drop3


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
def read_metadata(filename):
    # print('Reading metadata in file', filename)
    shapeParameterNames = []

    # Invoke the shell script and capture the output
    output = subprocess.check_output(['./getMetaData.sh', filename], shell=False)

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

    column_names = ['icl', 'Ecl']
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

    if useROOT:
        import ROOT
        print('\nProcessing file:', filename)
        print('Reading tree:', treename)
        df = ROOT.ROOT.RDataFrame("events", filename)
        num_init = df.Count()
        clusters = "AugmentedCaloClusters"
        df = df.Alias("clusters_energy", f"{clusters}.energy")
        df = (
            df
            .Define("icl", "ArgMax(clusters_energy)")
            .Define("Ecl", "clusters_energy[icl]")
            .Define("shapeParameters_begin_cl", f"{clusters}.shapeParameters_begin[icl]")
        )
        for iPar in shapeParamsIndex:
            df = df.Define(shapeParameterNames[iPar], f"_{clusters}_shapeParameters[shapeParameters_begin_cl + {iPar}]")

        d = df.Filter(f"icl>=0 && Ecl>{emin} && Ecl<{emax}")
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
                'AugmentedCaloClusters/AugmentedCaloClusters.energy',
                'AugmentedCaloClusters/AugmentedCaloClusters.shapeParameters_begin',
                'AugmentedCaloClusters/AugmentedCaloClusters.shapeParameters_end',
                '_AugmentedCaloClusters_shapeParameters',
            ],
            library='np')
        ecl = arr['AugmentedCaloClusters/AugmentedCaloClusters.energy']
        parBegin = arr['AugmentedCaloClusters/AugmentedCaloClusters.shapeParameters_begin']
        # parEnd = arr['AugmentedCaloClusters/AugmentedCaloClusters.shapeParameters_end']
        shapePars = arr['_AugmentedCaloClusters_shapeParameters']

        # loop over tfile events and fill dataframe
        nentries = len(ecl)
        print('Number of entries:', nentries)
        data = []
        skipped_entries = []
        for entry in tqdm(range(nentries)):
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
    outfile = outdir + '/plots/bdt-photonid-importances.pdf'
    print("\nPlotting the ranked feature importance to %s ...\n" % outfile)
    # If split, result contains numbers of times the feature is used in a model.
    # If gain, result contains total gains of splits which use the feature
    if trainingTool == 'lgbm':
        feature_imp = pd.DataFrame({'Value': model.feature_importance(importance_type=imptype), 'Feature': X.columns})
    elif trainingTool == 'xgboost':
        feature_important = model.get_score(importance_type=imptype)  # could also be weight
        values = list(feature_important.values())
        feature_imp = pd.DataFrame({'Value': values, 'Feature': X.columns})

    # debug
    # print(feature_imp)
    # fig = plt.figure(figsize=fig_size)
    fig = plt.figure()
    # sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Feature importance (%s)' % imptype)
    plt.tight_layout()
    plt.savefig(outfile)
    # plt.show()
    plt.close(fig)
    # save full list of feature importance to file
    feature_imp.sort_values(by='Value', ascending=False).to_csv(outdir + '/models/bdt-photonid-importances.csv', index=False, sep=" ")

# -------------------------------------------------------------------------------------------

# helper functions for 1D plots of input variables


def title(var):
    if var == 'Ecl':
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
    if var == 'Ecl':
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
    with PdfPages(outdir + '/plots/showershapes.pdf') as pdf:
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
                    fig_single.savefig(outdir + '/plots/%s.pdf' % varname)
                    plt.close(fig_single)

            plt.close(fig)


# -------------------------------------------------------------------------------------------


# Find the rows and columns of all missing values
def find_nan(df):
    # Reshape the DataFrame into a Series
    # series = df.stack(dropna=False)  # this will keep NaN - useful for debugging, but will raise a warning
    series = df.stack(future_stack=True)  # this wont keep rows with NaN
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
    # columns_to_drop3 = gamma_df.columns[gamma_df.columns.str.startswith('Ratio_E_max_2ndmax_vs_phi_EMB_layer')]
    columns_to_drop = ['icl'] + columns_to_drop1.tolist() + columns_to_drop2.tolist()
    # + columns_to_drop3.tolist()
    gamma_df = gamma_df.drop(columns_to_drop, axis=1)
    pi0_df = pi0_df.drop(columns_to_drop, axis=1)

    # can also do
    # gamma_df = gamma_df.dropna()
    # pi0_df = pi0_df.dropna()

    # check that there are no further NaN
    find_nan(gamma_df)
    find_nan(pi0_df)

    return gamma_df, pi0_df


# -------------------------------------------------------------------------------------------


# train and return a model given a dataframe
def trainBDT_LGB(df):

    # print the input dataset
    print(df)

    # define target variable (y) and input features (X)
    y = df['class']
    X = df.drop('class', axis=1)

    # split the input dataset in train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # scale input variables
    if scaleInputs:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Save the arrays to a .npz file
    np.savez(outdir + '/models/bdt-inputs-train-test-split.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, columns=X.columns)

    # loading data
    print("\nLoading the test and train datasets into LGB classifier...")
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
    else:
        params = {
            # 'task': 'train',
            'objective': 'binary',
            'boosting': 'gbdt',
            'metric': {'auc'},
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'max_bin': max_bin,
            # 'use_quantized_grad': True,
            'metric_freq': 1,
            'early_stopping_rounds': early_stopping_rounds,
            'num_iteration': epochs,
            'max_depth': max_depth,
            # 'min_data_in_bin': min_data_in_bin,  # or min_data_in_leaf?
            'feature_fraction': feature_fraction,
            'is_provide_training_metric': True,
            # 'num_threads': 24,
            'verbosity': 0,
        }
        outfile = outdir + '/models/trainingparams.json'
        file = open(outfile, 'w')
        json.dump(params, file, default=list)
        file.close()

    print('\nTraining parameters:')
    print(params)

    # fitting the model
    print("\nPerforming the training...")
    evals = {}
    model = lgb.train(params,
                      train_set=lgb_train,
                      valid_sets=[lgb_train, lgb_test],
                      valid_names=["train", "test"],
                      callbacks=[lgb.log_evaluation(), lgb.record_evaluation(evals)],
                      )

    # save model
    outfile = outdir + '/models/bdt-photonid.txt'
    print("\nSaving the model to file %s ..." % outfile)
    model.save_model(outfile)
    outfile = outdir + '/models/bdt-photonid.onnx'
    print("\nSaving the model to file %s ..." % outfile)
    model_onnx = onnxmltools.convert_lightgbm(model,
                                              initial_types=[('X', FloatTensorType([None, X_train.shape[1]]))],
                                              zipmap=False,
                                              split=100)
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

    # plot the history of the training
    outfile = outdir + '/plots/bdt-photonid-training-history-auc.pdf'
    print("\nDrawing the training history to %s ..." % outfile)
    lgb.plot_metric(evals, metric='auc')
    plt.savefig(outfile)
    # plt.show()

    return model, X_train, X_test, y_train, y_test


# -------------------------------------------------------------------------------------------


# Define the custom evaluation function (NOT USED)
def eval_auc(preds, dtrain):
    labels = dtrain.get_label()
    preds = preds.reshape(-1, 1)
    auc = ras(labels, preds)
    return 'auc', auc


# train and return a model given a dataframe
def trainBDT_XGB(df):

    # print the input dataset
    print(df)

    # define target variable (y) and input features (X)
    y = df['class']
    X = df.drop('class', axis=1)

    # rename features or otherwise export to onnx will fail
    new_column_names = {old_name: f'f{i}' for i, old_name in enumerate(X.columns)}
    X = X.rename(columns=new_column_names)
    print(X)

    # split the input dataset in train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # scale input variables
    if scaleInputs:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Save the arrays to a .npz file
    np.savez(outdir + '/models/bdt-inputs-train-test-split.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, columns=X.columns)

    # loading data
    print("\nLoading the test and train datasets into XGB classifier...")
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

    print("\nPerforming the training...")
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model = initial_model.fit(X_train, y_train,
                              eval_set=eval_set,
                              verbose=10)

    # Extract the AUC history from the evaluation results
    train_auc = model.evals_result()['validation_0']['auc']
    test_auc = model.evals_result()['validation_1']['auc']

    # plot the history of the training
    outfile = outdir + '/plots/bdt-photonid-training-history-auc.pdf'
    print("\nDrawing the training history to file %s ..." % outfile)
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
    outfile = outdir + '/models/bdt-photonid.json'
    print("\nSaving the model to file %s ..." % outfile)
    model.save_model(outfile)
    outfile = outdir + '/models/bdt-photonid.onnx'
    print("\nSaving the model to file %s ..." % outfile)
    model_onnx = onnxmltools.convert_xgboost(model,
                                             initial_types=[('X', FloatTensorType([None, X_train.shape[1]]))])
    onnx.save(model_onnx, outfile)

    return model, X_train, X_test, y_train, y_test

# -------------------------------------------------------------------------------------------


def optimiseBDT_XGB(df):

    # print the input dataset
    print(df)

    # define target variable (y) and input features (X)
    y = df['class']
    X = df.drop('class', axis=1)

    # rename features or otherwise export to onnx will fail
    # new_column_names = {old_name: f'f{i}' for i, old_name in enumerate(X.columns)}
    # X = X.rename(columns=new_column_names)
    # print(X)

    # split the input dataset in train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # scale input variables
    if scaleInputs:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Save the arrays to a .npz file
    # np.savez(outdir + '/models/bdt-inputs-train-test-split.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, columns=X.columns)

    # loading data and defining the model
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

    print("\nPerforming the training...")
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model = initial_model.fit(X_train, y_train,
                              eval_set=eval_set,
                              verbose=10)

    params = {
        'learning_rate': [0.01, 0.006, 0.003, 0.001],
        'min_child_weight': [1,3, 5,7, 10],
        'gamma': [0, 0.5, 1, 1.5, 2, 2.5, 5],
        'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5 , 6, 7, 8, 9, 10, 12, 14],
        'reg_lambda': np.array([0.4, 0.6, 0.8, 1, 1.2, 1.4])}

    # specific parameters. I set early stopping to avoid overfitting and specify the validation dataset
    fit_params = {
        #'early_stopping_rounds':10,
        'eval_set':[(X_test, y_test)]}


    # let's run the optimization
    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=500,
                                       scoring="roc_auc", n_jobs=-1,  verbose=3, random_state=42, cv=3  )
    random_search.fit(X_train, y_train, **fit_params)
    print(" Results from Random Search " )
    print("\n The best estimator across ALL searched params:\n", random_search.best_estimator_)
    print("\n The best score across ALL searched params:\n", random_search.best_score_)
    print("\n The best parameters across ALL searched params:\n", random_search.best_params_)




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
parser.add_argument('--indir', type=str, default='', help='The folder where the BDT to test is located (this option is only valide when skipTraining is enabled)')
parser.add_argument('--trainingParams', type=str, default='', help='An optional json file containing the parameters for the BDT training')

args = parser.parse_args()
emin = int(args.emin)
emax = int(args.emax)
outdir = args.outdir
overwriteDir = args.overwriteDir
# print(args.skipTraining)
doTraining = not args.skipTraining
indir = args.indir
trainingParams = args.trainingParams

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

    # read the metadata in the gamma and pi0 files and make sure they are the sam
    shapeParameterNames_gamma = read_metadata(basedir + gamma_filename)
    shapeParameterNames_pi0 = read_metadata(basedir + pi0_filename)
    assert shapeParameterNames_gamma == shapeParameterNames_pi0, 'shower shape decorations must be the same in the two files'
    shapeParameterNames = shapeParameterNames_gamma
    # write the list of parameter names used at training time to the models directory
    outfile = outdir + '/models/metadata.json'
    print('\nMetadata have been read correctly from the two files. Writing to output file', outfile)
    file = open(outfile, 'w')
    json.dump(shapeParameterNames, file)
    file.close()

    trainingVars = ['ecl']
    # trainingVars = []
    trainingVars.extend([x for x in shapeParameterNames if x not in columns_to_drop])
    outfile = outdir + '/models/bdt-inputs.json'
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

    # varsToUse = {
    #     'Delta_E_2ndmax_min_EMB_layer_3',
    #     'width_theta_5Bin_EMB_layer_3',
    #     'mass',
    #     'width_theta_7Bin_EMB_layer_3',
    #     'width_module_EMB_layer_2',
    #     'E_fr_side_pm4_EMB_layer_3',
    #     'width_theta_3Bin_EMB_layer_3',
    #     'width_module_EMB_layer_3',
    #     'energy_fraction_EMB_layer_2',
    #     'width_module_EMB_layer_4',
    #     'Ratio_E_max_2ndmax_EMB_layer_3',
    #     'E_fr_side_pm2_EMB_layer_3',
    #     'width_module_EMB_layer_5',
    #     'Ecl',
    #     'width_theta_EMB_layer_2',
    #     'maxcell_E_EMB_layer_2',
    #     'ncells',
    #     'Ratio_E_max_2ndmax_vs_phi_EMB_layer_1',
    #     'Ratio_E_max_2ndmax_vs_phi_EMB_layer_3',
    #     'energy_fraction_EMB_layer_1',
    #     'Ratio_E_max_2ndmax_vs_phi_EMB_layer_4',
    #     'Ratio_E_max_2ndmax_vs_phi_EMB_layer_2',
    #     'E_fr_side_pm3_EMB_layer_3',
    #     'Delta_E_2ndmax_min_vs_phi_EMB_layer_1',
    #     'Delta_E_2ndmax_min_vs_phi_EMB_layer_4',
    #     'maxcell_E_EMB_layer_1',
    #     'width_theta_9Bin_EMB_layer_2',
    #     'energy_fraction_EMB_layer_3',
    #     'width_theta_EMB_layer_4',
    #     'width_theta_EMB_layer_3',
    #     'Delta_E_2ndmax_min_vs_phi_EMB_layer_2',
    #     'width_theta_9Bin_EMB_layer_3',
    #     'Ratio_E_max_2ndmax_EMB_layer_4',
    #     'Ratio_E_max_2ndmax_vs_phi_EMB_layer_5',
    #     'width_module_EMB_layer_1',
    #     'Delta_E_2ndmax_min_EMB_layer_4',
    #     'Delta_E_2ndmax_min_vs_phi_EMB_layer_3',
    #     'width_module_EMB_layer_6',
    #     'Delta_E_2ndmax_min_EMB_layer_5',
    #     'maxcell_E_EMB_layer_4',
    #     'Ratio_E_max_2ndmax_EMB_layer_5',
    #     'Delta_E_2ndmax_min_vs_phi_EMB_layer_5',
    #     'Delta_E_2ndmax_min_EMB_layer_1',
    #     'maxcell_E_EMB_layer_0',
    #     'Ratio_E_max_2ndmax_vs_phi_EMB_layer_6',
    #     'energy_fraction_EMB_layer_4',
    #     'maxcell_E_EMB_layer_3',
    #     'Delta_E_2ndmax_min_EMB_layer_7',
    #     'width_theta_EMB_layer_5',
    #     'width_theta_9Bin_EMB_layer_5',
    # } - {'Ecl'}
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

    # do the training
#    if trainingTool == 'lgbm':
#        model, X_train, X_test, y_train, y_test = trainBDT_LGB(df)
#    elif trainingTool == 'xgboost':
#        model, X_train, X_test, y_train, y_test = trainBDT_XGB(df)
    optimiseBDT_XGB(df)
else:
    # Read the previously trained bdt
    if indir == '':
        indir = outdir
    if trainingTool == 'lgbm':
        model = lgb.Booster(model_file=indir + '/models/bdt-photonid.txt')
    elif trainingTool == 'xgboost':
        model = xgb.Booster(model_file=indir + '/models/bdt-photonid.json')
    # Read the data
    if os.path.isfile(indir + '/models/bdt-inputs-train-test-split.npz'):
        # Load the arrays from the .npz file
        data = np.load(indir + '/models/bdt-inputs-train-test-split.npz', allow_pickle=True)
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
    model = xgb.Booster(model_file=indir + '/models/bdt-photonid.json')
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
ax_single.hist(bdt_score_test_gamma, range=(0, 1), bins=50, alpha=0.5, label='$\\gamma$')
ax_single.hist(bdt_score_test_pi0, range=(0, 1), bins=50, alpha=0.5, label='$\\pi^{0}$')
ax_single.set_xlabel('BDT probability')
ax_single.legend(loc='upper right')
fig_single.savefig(outdir + '/plots/bdt-score-lin.pdf')
ax_single.semilogy()
fig_single.savefig(outdir + '/plots/bdt-score-log.pdf')
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
outfile = outdir + '/plots/bdt-photonid-roc.pdf'
print("\nDrawing the ROC curve to %s ..." % outfile)
plt.savefig(outfile)
plt.close(fig)

# round probabilities, for accuracy
y_train_pred = y_train_pred.round()
y_test_pred = y_test_pred.round()
print("Accuracy (train)", accuracy_score(y_train, y_train_pred))
print("Accuracy (test)", accuracy_score(y_test, y_test_pred))

# calculate and plot confusion matrix
outfile = outdir + '/plots/bdt-photonid-confusion-validation.pdf'
print("\nDrawing the confusion matrix on the validation sample to %s ..." % outfile)
cm = confusion_matrix(y_test, y_test_pred)
#                      labels=np.arange(len(channels)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[r"$\pi^{0}$", r"$\gamma$"])
disp.plot()
# plt.show()
disp.figure_.savefig(outfile)
disp.figure_.clear()

outfile = outdir + '/plots/bdt-photonid-confusion-norm-validation.pdf'
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
