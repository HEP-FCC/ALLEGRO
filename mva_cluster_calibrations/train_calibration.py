# -------------------------------------------------------------------------------------------
#
# train_calibration.py
#
# Train a BDT for e/gamma energy regression with the ALLEGRO ECAL
#
# Author: Giovanni Marchiori (giovanni.marchiori@cern.ch)
#
# -------------------------------------------------------------------------------------------


# Program settings

useGPU = True                                                  # use GPU or CPU
readInputsFromROOT = True                                      # read input features from root or from saved dataframe
readInputsFromCsv = False                                      # read input features from root or from saved dataframe
readInputsFromPkl = False                                      # read input features from root or from saved dataframe
saveDataToCsv = False                                          # save data to CSV
saveDataToPkl = True                                           # save data to pickle
csvpklDir = "../fullsim/run/test/training_reconstruction_smallSWclusters_noise"  # directory where we save/load data in csv/pkl format
doTraining = True                                              # if false, only plot Ereco/Etrue
saveModelToONNX = True                                         # export model (also) to onnx portable format
particle_PDG = 22                                              # PDGid of particle (not used - assumes particle with highest p is the good one)
# particle_PDG = 11                                            # PDGid of particle (not used - assumes particle with highest p is the good one)
useWeights = False                                             # give larger weight to lower energy particles (no effect..)
useExtraFeatures = True                                        # use also cluster theta, theta % deltaTheta, phi % deltaPhi as extra input features (no gain observed..)
treeName = 'events'                                            # name of TTree in ROOT files
clusterCollections = [                                         # collections for which we train the regression
    'EMBCaloClusters',
    'EMBCaloTopoClusters',
    'EMBCaloClustersWithNoise',
    'EMBCaloTopoClustersWithNoise'
]
# dictionary of input files: path, filename, and whether to use merged file or not
# in general I used to run directly over the merged file, but with very big productions
# and noise on I started to have some overflow issue. This can be worked around
# running on the unmerged files (usechain = True), though slower
inputFiles = {
    #"lowE": {
    #    # 0.1-1 GeV
    #    "basedir": "../fullsim/run/test/training_reconstruction_smallSWclusters_noise",
    #    "filename": "production_reconstruction_particle_gamma_lowE.root",
    #    "usechain": False,
    #},
    "midE": {
        # 1-100 GeV
        "basedir": "../../../fullsim/run/test/training_reconstruction_smallSWclusters_noise/root",
        "filename": "production_reconstruction_particle_gamma_jobid*.root",
        "usechain": True,
    },
    #"highE": {
    #    # 100-105 GeV
    #    "basedir": "../fullsim/run/test/training_reconstruction_smallSWclusters_noise",
    #    "filename": "production_reconstruction_particle_gamma_highE.root",
    #    "usechain": False,
    #}
}

# -------------------------------------------------------------------------------------------


# list of branches to read in each TTree for given cluster collection

def branchesToRead(clusters):
    return [
        'MCParticles/MCParticles.PDG',
        'MCParticles/MCParticles.generatorStatus',
        'MCParticles/MCParticles.momentum.x',
        'MCParticles/MCParticles.momentum.y',
        'MCParticles/MCParticles.momentum.z',
        f'Augmented{clusters}/Augmented{clusters}.energy',
        f'Augmented{clusters}/Augmented{clusters}.position.x',
        f'Augmented{clusters}/Augmented{clusters}.position.y',
        f'Augmented{clusters}/Augmented{clusters}.position.z',
        f'Augmented{clusters}/Augmented{clusters}.shapeParameters_begin',
        f'_Augmented{clusters}_shapeParameters'
    ]

# -------------------------------------------------------------------------------------------


def clusterType(clusters):
    if "Topo" in clusters:
        return "topo"
    else:
        return "sw"


# -------------------------------------------------------------------------------------------


# information to be retrieved about produced samples

inputDataSWClustering = {
    "detector": "",
    "inputCells": [],
    "samplingFractions": [],
    "addNoise": [],
    "addCrosstalk": [],
    "noiseMaps": [],
    "crosstalkMaps": [],
    "clusterSizeParams": [],
    "clusterEmin": None,
}


inputDataTopoClustering = {
    "detector": "",
    "inputCells": [],
    "samplingFractions": [],
    "addNoise": [],
    "addCrosstalk": [],
    "noiseMaps": [],
    "crosstalkMaps": [],
    "clusterSNThresholds": [],
    "clusterConnectBarrels": None,
    "clusterEmin": None,
}


import re


def parseString(s):
    if s.lower() == "true":
        return True

    if s.lower() == 'false':
        return False

    try:
        number = int(s)
        return number
    except ValueError:
        try:
            number = float(s)
            return number
        except ValueError:
            return s


# retrieve info about settings of an algorithm from the log file
# (assumes the configuration was printed in the log)
# temporary workaround when settings were not saved in or easily parsed
# from metadata. Should read from metadata instead

def getAlgSettingsFromLogFile(algName, logFileName):
    result_dict = {}
    pattern = re.compile(rf"\[k4run\] Option name: {re.escape(algName)}\.(\S+) (.+)")
    with open(logFileName, 'r') as file:
        for line in file:
            match = pattern.match(line)
            if match:
                key = match.group(1)
                value = parseString(match.group(2))
                result_dict[key] = value
    return result_dict


# get info about algorithms used to create cells and clusters
# in principle, info should be read from metadata in root file
# as a workaround, search for corresponding logfile and then extract
# info from there

def getClusterInfo(inputFile, clusters):
    # find logfile
    logFile = ""
    fileName = inputFiles[inputFile]["filename"].removesuffix(".root")
    if fileName.endswith("_jobid*"):
        fileName = fileName.removesuffix("_jobid*")
    fileName += "_jobid*.log"
    # print(fileName)
    baseDir = inputFiles[inputFile]["basedir"]
    # print(baseDir)
    if os.path.isdir(baseDir + "/log"):
        logFiles = glob.glob(baseDir + "log/" + fileName)
        if len(logFiles) > 0:
            logFile = logFiles[0]
            print("Reading information from logfile", logFile)
    elif os.path.isdir(baseDir + "/../log"):
        logFiles = glob.glob(baseDir + "/../log/" + fileName)
        if len(logFiles) > 0:
            logFile = logFiles[0]
            print("Reading information from logfile", logFile)
    else:
        print("Directory containing the log files not found!!!")
        exit(0)
    if logFile == "":
        print("Log file(s) not found!!!")
        exit(0)

    options = getAlgSettingsFromLogFile("CreatePositionedECalBarrelCells", logFile)
    options.update(getAlgSettingsFromLogFile("Create"+clusters, logFile))
    for key, value in options.items():
        print(f"{key}: {value}")


# -------------------------------------------------------------------------------------------


# imports

import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import scipy.stats as stats
if saveModelToONNX:
    import onnx
    import onnxmltools
if readInputsFromROOT:
    import uproot
import subprocess
from math import sqrt, acos, atan2, pi
from tqdm import tqdm
import glob

# -------------------------------------------------------------------------------------------


# some constants

deltaThetaCalo = 0.009817477 / 4
deltaPhiCalo = 2 * pi / 1536

# -------------------------------------------------------------------------------------------


# function for plotting the importance ranking of the input features

def plotImp(model, X, num=20, fig_size=(40, 20), clusters="clusters", imptype="gain", emin=0, emax=1000):
    print("\nPlotting the ranked feature importance...")
    # If split, result contains numbers of times the feature is used in a model.
    # If gain, result contains total gains of splits which use the feature
    feature_imp = pd.DataFrame({'Value': model.feature_importance(importance_type=imptype),
                                'Feature': X.columns})

    # plt.figure(figsize=fig_size)
    # sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('LightGBM Feature importance (%s)' % imptype)
    plt.tight_layout()
    outfile = f'plots/lgbm_importances-{clusters}-energy-{emin}-{emax}.pdf'
    plt.savefig(outfile)
    print("File %s produced" % outfile)
    # plt.show()

# -------------------------------------------------------------------------------------------


# function to read and return list of shower shape parameters from file

def read_metadata(filename, clusters):
    print('Reading metadata in file', filename)
    shapeParameterNames = []

    # Invoke the shell script and capture the output
    output = subprocess.check_output(['./getMetaData.sh', filename, clusters], shell=False)

    # Decode the output (it's a bytes object) and split it into lines
    lines = output.decode().splitlines()
    print(lines)

    for line in lines:
        shapeParameterNames.append(line.split()[1])
    return shapeParameterNames

# -------------------------------------------------------------------------------------------


# read the info in the numpy arrays in arr and fill the numpy array with particle momentum,
# cluster energy and other features needed for the calibration

def fillVectorsFromROOTBranches(arr, emin, emax, nLayers, inputFeaturePositions):
    px_part = arr['MCParticles/MCParticles.momentum.x']
    py_part = arr['MCParticles/MCParticles.momentum.y']
    pz_part = arr['MCParticles/MCParticles.momentum.z']
    # pdg_part = arr['MCParticles/MCParticles.PDG']
    # status_part = arr['MCParticles/MCParticles.generatorStatus']
    e_clusters = arr[f'Augmented{clusters}/Augmented{clusters}.energy']
    x_clusters = arr[f'Augmented{clusters}/Augmented{clusters}.position.x']
    y_clusters = arr[f'Augmented{clusters}/Augmented{clusters}.position.y']
    z_clusters = arr[f'Augmented{clusters}/Augmented{clusters}.position.z']
    parBegin = arr[f'Augmented{clusters}/Augmented{clusters}.shapeParameters_begin']
    shapeParameters = arr[f'_Augmented{clusters}_shapeParameters']

    # calculate particle energy
    nentries = len(px_part)
    p_part = np.zeros(nentries)
    # px_part, py_part, pz_part are lists of numpy arrays..
    for entry in range(nentries):
        # this selects photons/electrons (depending on PDG) with status=1
        # mask = (pdg_part[entry] == particle_PDG) & (status_part[entry] == 1)
        # p_part[entry] = (np.sqrt(px_part[entry][mask]**2 + py_part[entry][mask]**2 + pz_part[entry][mask]**2)).item()

        # this selects the highest momentum particle
        p = np.sqrt(px_part[entry]**2 + py_part[entry]**2 + pz_part[entry]**2)
        p_part[entry] = np.max(p)

    # find out highest energy cluster and retrieve its energy, energy fractions per layer, theta, phi
    e_cl = np.zeros(nentries)
    theta_cl = np.zeros(nentries)
    phi_cl = np.zeros(nentries)
    efrac_cl = []
    for layer in range(nLayers):
        efrac_cl.append(np.zeros(nentries))

    keep = np.ones(nentries, dtype=bool)
    for entry in range(nentries):
        if len(e_clusters[entry]) == 0:
            print('No clusters found for event %d, p=%f, skipping' % (entry, p_part[entry]))
            keep[entry] = 0
        else:
            icl = np.argmax(e_clusters[entry])
            e_cl[entry] = e_clusters[entry][icl]
            x_cl = x_clusters[entry][icl]
            y_cl = y_clusters[entry][icl]
            z_cl = z_clusters[entry][icl]
            r_cl = sqrt(x_cl**2 + y_cl**2 + z_cl**2)
            theta_cl[entry] = acos(z_cl / r_cl)
            phi_cl[entry] = atan2(y_cl, x_cl)
            for layer in range(nLayers):
                position = parBegin[entry][icl] + inputFeaturePositions[layer]
                efrac_cl[layer][entry] = shapeParameters[entry][position]
            # hack to see what happens if we sum up all clusters
            # e_cl[entry] = np.sum(e_clusters[entry])

            # keep only clusters in given energy range
            if e_cl[entry] < emin or e_cl[entry] > emax:
                keep[entry] = 0

            # drop events with poorly reconstructed e_true/e_cl (not in 0.4..1.7)
            if e_cl[entry] / p_part[entry] < 0.3 or e_cl[entry] / p_part[entry] > 1.7:
                keep[entry] = 0

    # remove events without clusters
    p_part = p_part[keep]
    e_cl = e_cl[keep]
    for i in range(nLayers):
        efrac_cl[i] = (efrac_cl[i])[keep]
    theta_cl = theta_cl[keep]
    phi_cl = phi_cl[keep]

    return p_part, e_cl, efrac_cl, theta_cl, phi_cl

# -------------------------------------------------------------------------------------------


# function to read an input file and fill a pandas dataframe with the relevant information

def readROOTFileIntoPandas(inputfile, clusters, emin, emax, nLayers):
    basedir = inputFiles[inputfile]["basedir"]
    filename = inputFiles[inputfile]["filename"]
    usechain = inputFiles[inputfile]["usechain"]
    filepath = os.path.abspath(basedir) + "/" + filename

    # find out positions of input features
    if not usechain:
        shapeParameterNames = read_metadata(filepath, f'Augmented{clusters}')
    else:
        # find one file matching the pattern and use it to read the metadata
        fileList = glob.glob(filepath)
        if fileList:
            shapeParameterNames = read_metadata(fileList[0], f'Augmented{clusters}')
        else:
            print("No files found matching the pattern", filepath)
            print("Exiting ...")
            exit(0)
    nLayersFromROOTFileShapeParams = sum(1 for s in shapeParameterNames if s.startswith("energy_fraction_EMB_layer_"))
    if nLayersFromROOTFileShapeParams != nLayers:
        print(f"Number of layers in shape parameters ({nLayersFromROOTFileShapeParams}) does not match that passed as parameter to train ({nLayers}), quitting..")
        exit(0)
    inputFeaturePositions = [-1] * nLayers
    for iLayer in range(nLayers):
        inputFeaturePositions[iLayer] = shapeParameterNames.index(f"energy_fraction_EMB_layer_{iLayer}")
    if -1 in inputFeaturePositions:
        print("Some input feature could not be found, quitting..")
        exit(0)
    print("All input features found in metadata, in positions:", inputFeaturePositions)

    # open root file and read events tree
    print("Reading events from ROOT file(s)", filepath)
    if not usechain:
        afile = uproot.open(filepath)
        events = afile[treeName]
        arr = events.arrays(branchesToRead(clusters), library='np')
        p_part, e_cl, efrac_cl, theta_cl, phi_cl = fillVectorsFromROOTBranches(arr, emin, emax, nLayers, inputFeaturePositions)
    else:
        p_part = np.array([])
        e_cl = np.array([])
        theta_cl = np.array([])
        phi_cl = np.array([])
        efrac_cl = []
        for layer in range(nLayers):
            efrac_cl.append(np.array([]))

        for arr in tqdm(uproot.iterate([filepath + ":" + treeName],
                                       branchesToRead(clusters), step_size="2 GB", library="np")):
            p_part_batch, e_cl_batch, efrac_cl_batch, theta_cl_batch, phi_cl_batch = fillVectorsFromROOTBranches(arr, emin, emax, nLayers, inputFeaturePositions)
            p_part = np.concatenate([p_part, p_part_batch])
            e_cl = np.concatenate([e_cl, e_cl_batch])
            theta_cl = np.concatenate([theta_cl, theta_cl_batch])
            phi_cl = np.concatenate([phi_cl, phi_cl_batch])
            for layer in range(nLayers):
                efrac_cl[layer] = np.concatenate([efrac_cl[layer], efrac_cl_batch[layer]])

    print('Total number of events selected:', len(p_part))
    print('')

    # fill numpy array with input features
    # - energy fractions
    # - possibly extra features related to cluster position
    # - total cluster energy
    efrac_layers = np.array([efrac_cl[i] for i in range(nLayers)])
    inputs = [efrac_layers]
    if useExtraFeatures:
        theta_cl_mod_calo = theta_cl / deltaThetaCalo % 1
        phi_cl_mod_calo = phi_cl / deltaPhiCalo % 1
        inputs.extend([theta_cl, theta_cl_mod_calo, phi_cl_mod_calo])
    inputs.append(e_cl)
    features = np.vstack(inputs)

    # calculate target
    target = p_part / e_cl

    # combine and import into pandas dataframe
    data = np.vstack((features, target.transpose())).transpose()
    df = pd.DataFrame(data=data[0:, 0:],
                      index=[i for i in range(data.shape[0])],
                      columns=['f' + str(i) for i in range(data.shape[1])])
    return df

# -------------------------------------------------------------------------------------------


# function to do the training

def train(clusters='EMBCaloClusters', emin=0, emax=1000, optimise=False, optType='sk-random', nLayers=11):

    print("\nTraining for cluster collection: %s" % clusters)

    # read info about clustering settings from log and ROOT files
    # read features and target from npy or ROOT files
    print("\nImporting features and target into pandas dataframe...")
    if readInputsFromROOT:
        df_list = []
        for i, inputFile in enumerate(inputFiles):
            # read info about clustering from metadata (and log)
            getClusterInfo(inputFile, clusters)
            df = readROOTFileIntoPandas(inputFile, clusters, emin, emax, nLayers)
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        # save the dataframe for later use
        if saveDataToCsv or saveDataToPkl:
            if os.access(csvpklDir, os.W_OK):
                if saveDataToCsv:
                    df.to_csv(csvpklDir + '/features-and-target-' + clusters + ".csv", index=False)
                if saveDataToPkl:
                    df.to_pickle(csvpklDir + '/features-and-target-' + clusters + ".pkl")
            else:
                print('WARNING: cannot write to directory', csvpklDir)
    elif readInputsFromCsv:
        filename = os.path.abspath(csvpklDir) + '/features-and-target-' + clusters + '.csv'
        print("Loading input features and target from file", filename)
        df = pd.read_csv(filename)
    elif readInputsFromPkl:
        filename = os.path.abspath(csvpklDir) + '/features-and-target-' + clusters + '.pkl'
        df = pd.read_pickle(filename)
    else:
        print('No input format specified, exiting')
        exit(0)

    # print some information about the dataset
    print('Total number of events in imported dataframe:', df.shape[0])
    print('')
    print(df.head())

    # calculate number of inputs
    nInputs = df.shape[1] - 1

    # define target variable
    y = df['f%d' % nInputs]

    # make a couple of plots
    fig, ax = plt.subplots()
    # ax.hist(y, range=(0.5, 1.5), bins=50, alpha=0.5, label='$E_{true}/E_{cl}$')
    ax.hist(y, range=(0.8, 1.2), bins=50, alpha=0.5, label=clusters)
    ax.set_xlabel('$E_{true}/E_{cl}$')
    ax.legend(loc='upper right')
    mean_value = np.mean(y)
    std_dev = np.std(y)
    ax.text(0.05, 0.9, '$\\mu = {:f}$'.format(mean_value), transform=ax.transAxes,
            fontsize=12, verticalalignment='top')
    ax.text(0.05, 0.85, '$\\sigma = {:f}$'.format(std_dev), transform=ax.transAxes,
            fontsize=12, verticalalignment='top')
    fig.savefig('plots/etrue_over_erec-training-%s.pdf' % clusters)
    plt.close(fig)

    # define feature variables
    X = df.drop('f%d' % nInputs, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # define weights for training and testing (weight more events with lower E)
    if useWeights:
        # should weight based on E_true = target * E_raw = y * X[nInputs]
        w_train = -np.log10(y_train * X_train.iloc[:, -1]) + 3
        # w_train = 1 * (X_train.iloc[:, -1] < 10)  # debug - only events with ecl<10 GeV
        w_test = -np.log10(y_test * X_test.iloc[:, -1]) + 3
        # w_train = w_train*w_train
        # w_test = w_test*w_test
        # w_train = 100./(y_train * X_train.iloc[:, -1])
        # w_test = 100./(y_test * X_test.iloc[:, -1])
        print(w_train)
        # for debug
        # if os.access(basedir, os.W_OK):
        #     X_train.to_csv(basedir + 'trainingdata-features-' + clusters + ".csv", index=False)
        #     np.savetxt(basedir + 'trainingdata-weights-' + clusters + ".csv", w_train, delimiter=",")

    if not doTraining:
        exit(0)

    # choose device to use for training
    if useGPU:
        device = 'gpu'
        # note: can set num_thread = 1 to reduce CPU usage when training on GPU
    else:
        device = 'cpu'

    # do the training or optimise the parameters
    if optimise and (optType == 'sk-random' or optType == 'sk-grid'):

        # grid search hyperparameter tuning with scikit-learn

        estimator = lgb.LGBMRegressor(objective='regression',
                                      boosting_type='gbdt',
                                      eval_metric='l2',
                                      random_state=101)

        if optType == 'sk-grid':
            # grid scan
            print("\nPerforming parameter optimisation with scikit-learn GridSearchCV\n")
            params = {
                'num_leaves': [nInputs + 1],
                'learning_rate': [0.05, 0.10, 0.15],
                'n_estimators': [500, 1000, 2000],
                'max_depth': [3, 7, 9, 11],
            }
            gsearch_lgb = GridSearchCV(estimator=estimator,
                                       param_grid=params,
                                       cv=3,  # cross-validation
                                       n_jobs=-1,  # parallel jobs (-1 = max)
                                       scoring='neg_mean_squared_error',
                                       # scoring='neg_root_mean_squared_error',
                                       verbose=10)
        else:
            # random search
            print("\nPerforming parameter optimisation with scikit-learn RandomizedSearchCV\n")
            params_scan_random = {'max_depth': stats.randint(3, nInputs),  # default 6
                                  'n_estimators': stats.randint(300, 800),  # default 100
                                  'learning_rate': stats.uniform(0.1, 0.5),  # def 0.3
                                  'subsample': stats.uniform(0.5, 1)}
            gsearch_lgb = RandomizedSearchCV(estimator=estimator,
                                             param_grid=params_scan_random,
                                             cv=3,  # cross-validation
                                             n_jobs=-1,  # parallel jobs (-1 = max)
                                             scoring='neg_mean_squared_error',
                                             # scoring='neg_root_mean_squared_error',
                                             niter=32,
                                             verbose=10)
        if useWeights:
            gsearch_lgb.fit(X_train, y_train, sample_weight=w_train)
        else:
            gsearch_lgb.fit(X_train, y_train)

        print('best params')
        print(gsearch_lgb.best_params_)
        preds_lgb_model = gsearch_lgb.predict(X_test)
        rmse_lgb = np.sqrt(mean_squared_error(y_test, preds_lgb_model))
        print("RMSE: %f" % (rmse_lgb))

    elif optimise and optType == 'lgb-manual':

        # grid search hyperparameter tuning using LGBM API

        print("\nPerforming manual parameter optimisation with randomized choice of LBGM training parameters\n")
        # defining default parameters
        params = {
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression',
            'num_leaves': nInputs + 1,
            'learning_rate': 0.1,
            'max_bin': 1023,
            # 'use_quantized_grad': True,
            'metric': {'l2', 'l1'},
            # 'metric': {'l2'},
            'early_stopping_rounds': 30,
            'num_iteration': 2000,
            # 'verbose': -1
        }

        iterations = 50
        minscore = 9999999999999.
        pp = {}
        for i in range(iterations):
            print('iteration number ', i)

            params['learning_rate'] = np.random.uniform(0., 0.05)
            # params['boosting'] = np.random.choice(['gbdt', 'dart'])
            params['max_bin'] = np.random.choice([255, 511, 1023, 2047])
            params['num_iteration'] = np.random.randint(100, 10000)
            params['early_stopping_rounds'] = np.random.randint(10, 100)
            print(params)

            # loading data
            if useWeights:
                lgb_train = lgb.Dataset(X_train, y_train, weight=w_train)
                lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, weight=w_test)
            else:
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

            # fitting the model
            model = lgb.train(params,
                              train_set=lgb_train,
                              valid_sets=lgb_eval)

            # evaluate model on test set and calculate MSE
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse**(0.5)
            print("MSE: %.6f" % mse)
            print("RMSE: %.6f" % rmse)

            # store params if MSE is minimum so far
            if mse < minscore:
                minscore = mse
                pp = params
        print('*' * 50)
        print('Minimum is ', minscore)
        print('Params: ', pp)

    elif not optimise:

        # training

        # load data in memory, defining training and testing samples
        print("\nLoading the test and train datasets into LGB regressor...")
        if useWeights:
            lgb_train = lgb.Dataset(X_train, y_train, weight=pd.Series(w_train))
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, weight=pd.Series(w_test))
        else:
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # defining parameters (see https://lightgbm.readthedocs.io/en/latest/Parameters.html)
        params = {
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression_l1',
            # 'objective': 'regression_l2',
            # 'objective': 'huber',  # wont convert to onnx...
            # 'objective': 'mape',  # wont convert to onnx...
            # 'num_leaves': nInputs+1,
            'num_leaves': 60,
            'learning_rate': 0.01,
            # 'max_bin': 1023  # bin size 1023 cannot run on GPU
            # 'use_quantized_grad': True,
            'metric': {'l1', 'l2'},
            'early_stopping_rounds': 50,
            'num_iteration': 3000,
            # 'verbose': -1,
            'device': device,
            # might need this to avoid overfitting. Will tranfer to gpu multiple times
            # "feature_fraction": 0.9,
            # "bagging_fraction": 0.8,
            # "bagging_freq": 5,
        }

        # print the training settings
        print('\nTraining parameters:')
        print(params)

        # fit the model
        print("\nPerforming the training...")
        evals = {}
        model = lgb.train(params,
                          train_set=lgb_train,
                          valid_sets=[lgb_eval, lgb_train],
                          valid_names=["test", "train"],
                          callbacks=[lgb.log_evaluation(10), lgb.record_evaluation(evals)])

        # save the model
        outfile = f'lgbm_calibration-{clusters}-energy-{emin}-{emax}'
        print("\nSaving the model to file %s.txt ..." % outfile)
        model.save_model('models/' + outfile + '.txt')
        if saveModelToONNX:
            from skl2onnx.common.data_types import FloatTensorType
            model_onnx = onnxmltools.convert_lightgbm(model,
                                                      initial_types=[('X', FloatTensorType([None, X_train.shape[1]]))],
                                                      split=100)

            onnx.save(model_onnx, 'models/' + outfile + '.onnx')

        # plot the history of the training
        print("\nDrawing the training history...")
        for metric in params['metric']:
            fig, ax = plt.subplots()
            lgb.plot_metric(evals, metric=metric, ax=ax)
            plt.savefig(f'plots/training-history-{clusters}-energy-{emin}-{emax}-{metric}.pdf')
            plt.close(fig)

        # prediction and accuracy check
        print("\nCalculating the final model accuracy...")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**(0.5)
        print("MSE: %.6f" % mse)
        print("RMSE: %.6f" % rmse)
        mae = mean_absolute_error(y_test, y_pred)
        print("MAE: %.6f" % mae)

        # plot the feature importance
        plotImp(model, X, clusters=clusters, emin=emin, emax=emax)
    else:
        print('Wrong options')

    print("\nDone\n")

# hyperparameter optimisation
# train('CaloClusters', True, 'lgb-manual')


# training
for clusters in clusterCollections:
    train(clusters, 0, 1000, False, '')
