# -------------------------------------------------------------------------------------------
#
# optimise.py
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
import uproot
import numpy as np
import json
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------------------------

# settings

debug = False
outdir = 'optimisation2'
overwriteDir = True

# initial_var_list = {'Ecl'}
# initial_var_list = {'Delta_E_2ndmax_min_EMB_layer_3', 'width_module_EMB_layer_3', 'width_module_EMB_layer_2', 'width_theta_5Bin_EMB_layer_3', 'width_theta_9Bin_EMB_layer_3', 'Ecl'}
# initial_auc = 0.919

# default training settings
learning_rate = 0.01
num_leaves = 63
max_bin = 255
early_stopping_rounds = 30
epochs = 100
feature_fraction = 0.9
# a json file that might override the training settins
trainingParams = ''

# the folder with the pi0 and gamma root files, the file and tree names
basedir = '/home/lit/public/giovanni/sample/photonID_stripL3/production_reconstruction/'
# basedir = os.getcwd() + '/'
gamma_filename = "production_reconstruction_particle_gamma.root"
pi0_filename = "production_reconstruction_particle_pi0.root"
treename = 'events'
shapeParameterNames = []  # to be readout from the files
shapeParams = set()

auc = []
varlist = []

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
def read_events(filename, treename, bdtclass, **kwargs):
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

    # open file
    print('\nProcessing file:', filename)
    afile = uproot.open(filename)

    # read tree
    print('Reading tree:', treename)
    tree = afile[treename]

    # read branches
    arr = tree.arrays([
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
    for entry in range(nentries):
    # for entry in range(200):
        if len(ecl[entry]) == 0:
            print('No clusters found for event %d, skipping' % entry)
        else:
            icl = np.argmax(ecl[entry])
            e = ecl[entry][icl]
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

    # create pandas dataframe from data (list of lists)
    df = pd.DataFrame(data, columns=column_names)
    df['class'] = bdtclass
    print('Dataframe content:')
    print(df)
    return df

# -------------------------------------------------------------------------------------------



# Find the rows and columns of all missing values
def find_nan(df):
    # Reshape the DataFrame into a Series
    series = df.stack(dropna=False)
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
    columns_to_drop1 = gamma_df.columns[gamma_df.columns.str.startswith('Ratio_E_max_2ndmax_vs_phi_EMB_layer')]
    columns_to_drop2 = gamma_df.columns[gamma_df.columns.str.startswith('theta_EMB_layer')]
    columns_to_drop3 = gamma_df.columns[gamma_df.columns.str.startswith('phi_EMB_layer')]
    # extra columns to drop (no separation power)]
    # columns_to_drop4 = ['E_fr_side_pm2_EMB_layer_{:d}'.format(i) for i in [0,1,2,4,5,6,7,8,9,10]]
    # columns_to_drop5 = ['E_fr_side_pm3_EMB_layer_{:d}'.format(i) for i in [0,1,2,4,5,6,7,8,9,10]]
    # columns_to_drop6 = ['width_theta_3Bin_EMB_layer_{:d}'.format(i) for i in [0,1,2,4,5,6,7,8,9,10]]
    # columns_to_drop7 = ['width_theta_5Bin_EMB_layer_{:d}'.format(i) for i in [0,1,2,4,5,6,7,8,9,10]]
    # columns_to_drop8 = ['width_theta_7Bin_EMB_layer_{:d}'.format(i) for i in [0,1,2,4,5,6,7,8,9,10]]

    columns_to_drop = ['icl'] + columns_to_drop1.tolist() + columns_to_drop2.tolist() + columns_to_drop3.tolist() + columns_to_drop4 + columns_to_drop5 + columns_to_drop6 + columns_to_drop7 + columns_to_drop8

    gamma_df = gamma_df.drop(columns_to_drop, axis=1)
    pi0_df = pi0_df.drop(columns_to_drop, axis=1)

    # check that there are no further NaN
    find_nan(gamma_df)
    find_nan(pi0_df)

    return gamma_df, pi0_df

# -------------------------------------------------------------------------------------------


# train and return a model given a dataframe
def trainBDT(df_train, df_test, variables):

    # define target variable (y) and input features (X)
    y_test = df_test['class']
    y_train = df_train['class']
    X_test = df_test.loc[:, df_test.columns.intersection(variables)]
    X_train = df_train.loc[:, df_train.columns.intersection(variables)]
    # print(X_train)
    # print(y_train)
    # print(X_test)
    # print(y_test)

    # loading data
    print("Loading the test and train datasets into LGB regressor...")
    # No weights
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
    # lgb_train = lgb.Dataset(x_train, y_train, weight=w_train)
    # lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train, weight=w_test)

    # defining parameters
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
        # 'num_iteration': epochs,
        # 'max_depth': 20,
        'feature_fraction': feature_fraction,
        'is_provide_training_metric': True,
        # 'num_threads': 24,
        # 'verbosity': 0,
        'verbosity': -1,
    }

    # print('\nTraining parameters:')
    # print(params)

    # fitting the model
    print("Performing the training...")
    evals = {}
    model = lgb.train(params,
                      train_set=lgb_train,
                      valid_sets=[lgb_train, lgb_test],
                      valid_names=["train", "test"],
                      num_boost_round = epochs,
                      # callbacks=[lgb.log_evaluation(), lgb.record_evaluation(evals)],
                      )
    score = model.best_score['test']['auc']
    # print(score)
    return score, model


def optimiseBDT(niter, df_train, df_test):
    print()
    print("-" * 120)
    print("\nIteration:", niter)
    if niter==0:
        # varlist_from_previous_iter = set()
        # vars_to_try={'Ecl'}
        # best_auc = 0.5
        # vars_to_try = initial_var_list.copy()
        varlist_from_previous_iter = {'Delta_E_2ndmax_min_EMB_layer_3', 'width_module_EMB_layer_3', 'width_module_EMB_layer_2', 'width_theta_5Bin_EMB_layer_3', 'width_theta_9Bin_EMB_layer_3', 'Ecl', 'energy_fraction_EMB_layer_2', 'width_module_EMB_layer_4'}
        best_auc = 0.9323
        vars_to_try = shapeParams - varlist_from_previous_iter

        best_auc_from_prev_iter = best_auc
    else:
        varlist_from_previous_iter = varlist[niter-1]
        best_auc = auc[niter-1]
        best_auc_from_prev_iter = auc[niter-1]
        vars_to_try = shapeParams - varlist_from_previous_iter
    initialShapeParams = shapeParams.copy()
    nvars = len(vars_to_try)
    print("Variables to be included:")
    print(varlist_from_previous_iter)
    print("Variables to try:", nvars)
    print(vars_to_try)
    best_var = ''
    best_model = None
    for var in vars_to_try:
        print('\nTrying var:', var)
        varlist_for_this_iter = varlist_from_previous_iter.union({var})
        _auc, _model = trainBDT(df_train, df_test, varlist_for_this_iter)
        print('Score:',_auc)
        # keep only variables that improve AUC by at least 0.15%
        if _auc > best_auc and _auc > best_auc_from_prev_iter*1.0015:
            best_auc = _auc
            best_var = var
            best_model = _model
        # remove from further optimisation the variables that did not lead to
        # significant improvements in auc
        if _auc < best_auc_from_prev_iter*1.0015:
            shapeParams.remove(var)
            print("Dropping Var:", var)
    if best_var!='':
        print("\nBest variable to be added:", best_var)
        print("Best AUC:", best_auc)
        if niter>0:
            print("AUC gain:", best_auc/auc[niter-1])
        auc.append(best_auc)
        varlist_for_next_iter = varlist_from_previous_iter.union({best_var})
        varlist.append(varlist_for_next_iter)
    else:
        print("\nNo variable to add found")
    print("Dropped variables:", initialShapeParams-shapeParams)
    return (best_var, best_model)

# -------------------------------------------------------------------------------------------


# main program

# parser = argparse.ArgumentParser(description='Optimise BDT for photon/pi0 discrimination')
# parser.add_argument('--outdir', type=str, default='inclusive', help='The output folder (optional)')
# parser.add_argument('--skipTraining', action='store_true', help='pass this option to skip the training')
# parser.add_argument('--overwriteDir', action='store_true', help='pass this option to overwrite an existing output directory during the training')
# parser.add_argument('--indir', type=str, default='', help='The folder where the BDT to test is located (this option is only valide when skipTraining is enabled)')
# parser.add_argument('--trainingParams', type=str, default='', help='An optional json file containing the parameters for the BDT training')

# args = parser.parse_args()
# outdir = args.outdir
# overwriteDir = args.overwriteDir
# # print(args.skipTraining)
# doTraining = not args.skipTraining
# indir = args.indir
# trainingParams = args.trainingParams

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
outfile = outdir + '/models/shapeParameters.json'
print('\nMetadata have been read correctly from the two files. Writing to output file', outfile)
file = open(outfile, 'w')
json.dump(shapeParameterNames, file)
file.close()

# read the photon and pi0 files into pandas dataframes
# add the column with the class (0 for bkg, 1 for signal) for the BDT
# print the content of the datasets
gamma_df = read_events(basedir + gamma_filename, treename, 1)
pi0_df = read_events(basedir + pi0_filename, treename, 0)

# example of how to read only some variables:
# gamma_df = read_events(basedir + gamma_filename, treename, 1, shapeParamsIndex=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# pi0_df = read_events(basedir + pi0_filename, treename, 0, shapeParamsIndex=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# pi0_df = read_events(pi0_filename, treename, shapeParamsName=['maxcell_E_EMB_layer_1', 'Delta_E_2ndmax_min_vs_phi_EMB_layer_8', 'width_theta_EMB_layer_10'])

# check if there are NaN
find_nan(gamma_df)
find_nan(pi0_df)

# drop all columns for Ratio_E_max_2ndmax_vs_phi: a lot of NaN
# sanity check: drop theta and phi of cluster barycenter if they were included in initial list of features, as well as cluster index
gamma_df, pi0_df = clean_dfs(gamma_df, pi0_df)

# create the combined dataset
df = pd.concat([gamma_df, pi0_df], ignore_index=True)
column_names = df.columns[:-1]

# split the dataset into test and train
df_train, df_test = train_test_split(df, test_size=0.15, random_state=42)
print(df_train)
print(df_test)


# do the optimisation
niter = 0
maxiter = 20
shapeParams = set(column_names)
while niter < maxiter:
    if niter>0 and len(shapeParams - varlist[niter-1])==0:
        break
    else:
        best_var, model = optimiseBDT(niter, df_train, df_test)
        if best_var!='':
            niter += 1
        else:
            break

print()
print("-" * 120)
print("\nOptimisation completed after %d iterations" % niter)
print("Chosen variables:")
print(varlist[niter-1])
print("AUC:", auc[niter-1])

print("\nHistory:")
for i in range(niter):
    print("{:02d} {:.3f} ".format(i, auc[i]), end='')
    print(varlist[i])

outfile = outdir + '/models/bdt-photonid.txt'
print("\nSaving the model to file %s ..." % outfile)
model.save_model(outfile)

# calculate probabilities, for AUC
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)
# test_auc = ras(y_test, y_test_pred)
# train_auc = ras(y_train, y_train_pred)
# print("Training ROC-AUC:", train_auc)
# print("Test ROC-AUC:", test_auc)


# print bkg eff at some benchmark sig eff
# get_bkg_eff(0.8, tpr, fpr)
# get_bkg_eff(0.9, tpr, fpr)
# get_bkg_eff(0.95, tpr, fpr)
# get_bkg_eff(0.99, tpr, fpr)
print("\n")
