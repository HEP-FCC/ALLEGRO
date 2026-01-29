#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------------------------------
#
# test_calibration.py
#
# Assess the performance of the e/gamma energy regression BDT on samples at fixed energy
#
# One can use different trainings for different intervals of cluster energy
# Currently the code is not optimised (the inference will be run on all events for each
# model and then the one corresponding to the cluster energy of the event will be chosen.
#
# Author: Giovanni Marchiori (giovanni.marchiori@cern.ch)
#
# -------------------------------------------------------------------------------------------


# Program settings

debug = False                                            # print debugging statements / make more plots
nLayers = 11                                             # number of longitudinal layers in calo
useParamsFromFits = True                                 # compute mu, sigma from stddev in data or from gaussian fit
readDataFromJson = True                                  # if true, will read the energies, responses and resolutions from json
                                                         # a json file rather than recomputing them from the ROOT files
                                                         # that requires running the inference
doNoise = True                                           # include noise term b/E in fit to resolution vs E
#doNoise = False                                         # include noise term b/E in fit to resolution vs E
clusterCollections = [                                   # collections for which we test the performance
    'EMBCaloClusters',
    'EMBCaloTopoClusters',
    'EMBCaloClustersWithNoise',
    'EMBCaloTopoClustersWithNoise'
]
# label = "EMB_topo_withnoise"
label = "EMB_calo_topo_w_wo_noise"
calibrationFiles = [
    'EMBCaloClusters',
    'EMBCaloTopoClusters',
    'EMBCaloClustersWithNoise',
    'EMBCaloTopoClustersWithNoise'
]
#basedir = "../fullsim/run/test/clusters/"               # directory where the input files are
#basedir = "../fullsim/run/test/clusters_with_noise/"     # directory where the input files are
#basedir = "../fullsim/run/test/clusters_nothreshold_topo/"     # directory where the input files are
basedir = "../fullsim/run/test/clusters_smallSWcluster/"     # directory where the input files are
particle = 'gamma'                                       # particle type
# particle = 'e-'                                        # particle type
useShapeParameters = True                                # read energy fraction per layer from shapeParameters or calculate it from cluster cell collection
energies = [                                             # energy points (in MeV)
    300,
    500,
    1000,
    5000,
    10000,
    15000,
    20000,
    30000,
    50000,
    75000,
    100000
]
useAK = False                                            # true for awkward, false for numpy
modelFormat = 'onnx'                                    # use model stored in onnx or lgbm format
#modelFormat = 'lgbm'                                     # use model stored in onnx or lgbm format
useExtraFeatures = True                                  # use also cluster theta, theta % deltaTheta, phi % deltaPhi as extra input features

#emin=0
#emax=10

calibrations = [
#    {
#        "emin_cl":0,
#        "emax_cl":15,
#        "emin_part":0,
#        "emax_part":15
#    },
    {
        "emin_cl":0,
        "emax_cl":1000,
        "emin_part":0,
        "emax_part":1000
    },
]

# -------------------------------------------------------------------------------------------

# some constants
deltaThetaCalo = 0.009817477/4
from math import pi
deltaPhiCalo = 2*pi/1536

# -------------------------------------------------------------------------------------------

# imports

import numpy as np
import matplotlib.pyplot as plt
# use this on linux
fontset = "Nimbus Sans"
# use this on mac
# fontset = "Helvetica"
plt.rcParams["font.family"] = fontset
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = fontset
plt.rcParams["mathtext.it"] = fontset+":italic"
plt.rcParams["mathtext.bf"] = fontset+":bold"
#
plt.rcParams["font.size"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
import uproot
import numpy as np
if useAK:
    import awkward as ak
from math import sqrt, atan2, acos, exp, log, pi, ceil
import hist
from numba import njit, jit
from scipy import optimize as opt
from scipy.stats import norm
import pickle
from io import BytesIO
import PIL.Image
import os
if modelFormat == 'onnx':
    import onnxruntime
elif modelFormat == 'lgbm':
    import lightgbm as lgb
else:
    print('Error: model format %s unknown' % modelFormat)
    import sys
    sys.exit()
import subprocess
from math import sqrt, acos, atan2
import json

# -------------------------------------------------------------------------------------------

# Helper functions

# -------------------------------------------------------------------------------------------

# Define energy resolution curves
def resol_curve_with_d(x, b, c, d):
    return np.sqrt(c*c + b*b/x+ d*d/np.sqrt(x))

def resol_curve(x, a, b, c):
    return np.sqrt(c*c + b*b/x + a*a/(x*x))

def resol_curve_no_noise(x, b, c):
    return resol_curve(x, 0.0, b, c)

def resol_curve_vs_invsqrtE(x, a, b, c):
    return np.sqrt(c*c + b*b*x*x + a*a*x*x*x*x)

def resol_curve_vs_invsqrtE_no_noise(x, b, c):
    return resol_curve_vs_invsqrtE(x, 0.0, b, c)

# Define the Gaussian function
def gauss(x, a, b, c):
    return (a/(sqrt(2*pi)*c))*np.exp(-0.5*((x-b)/c)**2)
    
# -------------------------------------------------------------------------------------------

# Define function to calculate energy per layer and compile it with numba
# starting from cell collections. Not needed if the fractional energies
# per layer are already saved as shapeParameters of the clusters
def calc_cluster_energy_per_layer(ecv, cidv, cell_begin, cell_end, nLayers):

    # initialize to zero the vectors of energy per layer
    ec = np.zeros(nLayers, dtype=np.float32)

    # retrieve vectors of cells e and cellid for given cluster
    _e = ecv[cell_begin:cell_end]
    _id = cidv[cell_begin:cell_end]

    # calculate the vector with the cell layers
    mask =  np.full(len(_e), ((1<<8) - 1) , dtype=np.uint64)
    nbits = np.full(len(_e), 11, dtype=np.uint64)
    layerv = np.bitwise_and(mask, np.right_shift(_id, nbits))
    #if debug:
    #    print(layerv)

    # loop over the layers
    for layer in range(nLayers):
        # for each layer, select the cells belonging to that layers and put them in dedicated arrays
        layer_mask = layerv == layer    
        layer_ce = _e[layer_mask]
        ec[layer] = np.sum(layer_ce)
    return ec

c_calc_cluster_energy_per_layer = njit(calc_cluster_energy_per_layer)

# -------------------------------------------------------------------------------------------

# function to read and return list of shower shape parameters from file
def read_metadata(filename, clusters):
    print('Reading metadata in file', filename)
    shapeParameterNames = []

    # Invoke the shell script and capture the output
    output = subprocess.check_output(['./getMetaData.sh', filename, clusters], shell=False)

    # Decode the output (it's a bytes object) and split it into lines
    lines = output.decode().splitlines()

    for line in lines:
        shapeParameterNames.append(line.split()[1])
    return shapeParameterNames

# -------------------------------------------------------------------------------------------

# function to read events for given particle energy
inputFeaturePositions = {}
def read_file(energy, particle, clusters, cells):
    library = 'np'
    if useAK:
        library = 'ak'

    if not energy in inputFeaturePositions:
        inputFeaturePositions[energy] = {}

    filename = "reconstruction_energy_%d_theta_90_particle_%s.root" % (energy, particle)
    fileWithPath = os.path.abspath(basedir) + "/" + filename
    treename = 'events'
    print('Reading file %s' % fileWithPath)
    f = uproot.open(fileWithPath)
    if debug:
        print('File contains the following keys:')
        print(f.keys())
    print('Reading tree %s' % treename)
    events = f[treename]
    if debug:
        print('Branches of tree %s:' % treename)
        events.show()

    branches = [
        'MCParticles/MCParticles.PDG',
        'MCParticles/MCParticles.generatorStatus',
        'MCParticles/MCParticles.momentum.x',
        'MCParticles/MCParticles.momentum.y',
        'MCParticles/MCParticles.momentum.z'
    ]
    if useShapeParameters:
        branches += [
            f'Augmented{clusters}/Augmented{clusters}.energy',
            f'Augmented{clusters}/Augmented{clusters}.position.x',
            f'Augmented{clusters}/Augmented{clusters}.position.y',
            f'Augmented{clusters}/Augmented{clusters}.position.z',
        ]
    else:
        branches += [
            clusters+'/'+clusters+'.energy',
            clusters+'/'+clusters+'.position.x',
            clusters+'/'+clusters+'.position.y',
            clusters+'/'+clusters+'.position.z'
        ]
    if useShapeParameters:
        branches += [
            f'Augmented{clusters}/Augmented{clusters}.shapeParameters_begin',
            f'_Augmented{clusters}_shapeParameters'
            ]
        if not clusters in inputFeaturePositions[energy]:
            # initialise list of input feature positions, assuming it's the same for each file
            shapeParameterNames = read_metadata(fileWithPath, f'Augmented{clusters}')
            nLayersFromROOTFileShapeParams = sum(1 for s in shapeParameterNames if s.startswith("energy_fraction_EMB_layer_"))
            if nLayersFromROOTFileShapeParams != nLayers:
                print(f"Number of layers in shape parameters ({nLayersFromROOTFileShapeParams}) does not match nLayers ({nLayers}), quitting..")
                exit(0)
            inputFeaturePositions[energy][clusters] = [-1] * nLayers
            for iLayer in range(nLayers):
                inputFeaturePositions[energy][clusters][iLayer] = shapeParameterNames.index(f"energy_fraction_EMB_layer_{iLayer}")
            if -1 in inputFeaturePositions[energy][clusters]:
                print("Some input feature could not be found, quitting..")
                exit(0)
            print("All input features found in metadata, in positions:", inputFeaturePositions[energy][clusters])

    else:
        branches += [
            clusters+'/'+clusters+'.hits_begin',
            clusters+'/'+clusters+'.hits_end',
            cells+'/'+cells+'.cellID',
            cells+'/'+cells+'.energy',
        ]

    arr = events.arrays(
        branches,
        library=library)    
    return arr

# -------------------------------------------------------------------------------------------

# Load lightgbm model
def readCalibration(clusters, emin=0, emax=1000):
    model_file = f'models/lgbm_calibration-{clusters}-energy-{emin}-{emax}.txt'
    print("Reading calibration parameters from file:", model_file)
    return lgb.Booster(model_file=model_file)

# -------------------------------------------------------------------------------------------

def fig2img(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img

# -------------------------------------------------------------------------------------------

#
# beginning of program
#

resolutions_raw = {}
resolutions_cal = {}
resolutions_cal_err = {}
responses_raw = {}
responses_cal = {}
fitparams = {}

if readDataFromJson:
    jsonFileName = label+'.json'
    print("Reading energies, responses and resolutions from JSON file", jsonFileName)
    with open(jsonFileName, 'r') as jsonFile:
        dataFromJson = json.load(jsonFile)
        energies = dataFromJson["energies"]
        resolutions_raw = dataFromJson["resolutions_raw"]
        resolutions_cal = dataFromJson["resolutions_cal"]
        resolutions_cal_err = dataFromJson["resolutions_cal_err"]
        responses_raw = dataFromJson["responses_raw"]
        responses_cal = dataFromJson["responses_cal"]
    if debug:
        print(energies)
        print(resolutions_raw)
        print(resolutions_cal)
        print(responses_raw)
        print(responses_cal)

energies_in_gev = [x/1000 for x in energies]
invsqrtenergies = [1/sqrt(x/1000.) for x in energies]

# calculate responses and resolutions from ROOT files if not from json:
if not readDataFromJson:
    ncalibrations = len(calibrations)
    print("Number of calibrations:", ncalibrations)

    # loop over cluster collections to fill arrays of resolution and response
    # vs energy for each cluster collection
    print('Processing the inputs, making basic plots and applying the calibration...')
    for index, clusters in enumerate(clusterCollections):

        print('\nCluster collection = %s' % clusters)

        # initialise arrays that will contain the resolution and response
        resolutions_raw[clusters] = [0]*len(energies)
        resolutions_cal[clusters] = [0]*len(energies)
        resolutions_cal_err[clusters] = [0]*len(energies)
        responses_raw[clusters] = [0]*len(energies)
        responses_cal[clusters] = [0]*len(energies)

        # determine corresponding cell collection name (needed only if clusters are not decorated)
        cells = clusters[:-1]+'Cells'

        # read LGBM models
        model = []
        session = []
        if modelFormat=='lgbm':
            for i in range(ncalibrations):
                emin = calibrations[i]["emin_part"]
                emax = calibrations[i]["emax_part"]
                print(f"Model {i}: emin={emin}, emax={emax}")
                model.append(readCalibration(calibrationFiles[index], emin, emax))
        else:
            for i in range(ncalibrations):
                emin = calibrations[i]["emin_part"]
                emax = calibrations[i]["emax_part"]
                calibFile = calibrationFiles[index]
                model_file = f'models/lgbm_calibration-{calibFile}-energy-{emin}-{emax}.onnx'
                print("Reading calibration parameters from file:", model_file)
                session.append(onnxruntime.InferenceSession(model_file,
                                                            providers=["CPUExecutionProvider"]))

        # loop over energies
        idx = 0
        for energy in energies:
            print('\nEnergy = %.2f GeV' % (energy/1000))

            # open file and read tree
            arr = read_file(energy, particle, clusters, cells)

            pdg_part = arr['MCParticles/MCParticles.PDG']
            status_part = arr['MCParticles/MCParticles.generatorStatus']
            px_part = arr['MCParticles/MCParticles.momentum.x']
            py_part = arr['MCParticles/MCParticles.momentum.y']
            pz_part = arr['MCParticles/MCParticles.momentum.z']

            if useShapeParameters:
                e_clusters = arr[f'Augmented{clusters}/Augmented{clusters}.energy']
                x_clusters = arr[f'Augmented{clusters}/Augmented{clusters}.position.x']
                y_clusters = arr[f'Augmented{clusters}/Augmented{clusters}.position.y']
                z_clusters = arr[f'Augmented{clusters}/Augmented{clusters}.position.z']
            else:
                e_clusters = arr[clusters+'/'+clusters+'.energy']
                x_clusters = arr[clusters+'/'+clusters+'.position.x']
                y_clusters = arr[clusters+'/'+clusters+'.position.y']
                z_clusters = arr[clusters+'/'+clusters+'.position.z']
            
            if useShapeParameters:
                parBegin = arr[f'Augmented{clusters}/Augmented{clusters}.shapeParameters_begin']
                shapeParameters = arr[f'_Augmented{clusters}_shapeParameters']
            else:
                firstcell_clusters = arr[clusters+'/'+clusters+'.hits_begin']
                lastcell_clusters = arr[clusters+'/'+clusters+'.hits_end']
                id_cells = arr[cells+'/'+cells+'.cellID']
                e_cells = arr[cells+'/'+cells+'.energy']
        
            nentries = len(px_part)
            print('Entries = %d' % nentries)

            # uses previous function to calculate energy of cluster icl in event entry
            # by retrieving the cell energy vectors in event entry
            # and the first/last cells of cluster icl of the same event
            # and passing them to the compiled function above

            if not useShapeParameters:
                def calc_energy_per_layer_ak(entry, icl):
                    return c_calc_cluster_energy_per_layer(e_cells[entry].to_numpy(),
                                                           id_cells[entry].to_numpy(),
                                                           firstcell_clusters[entry][icl],
                                                           lastcell_clusters[entry][icl], nLayers)
        
                def calc_energy_per_layer(entry, icl):
                    return c_calc_cluster_energy_per_layer(e_cells[entry],
                                                           id_cells[entry],
                                                           firstcell_clusters[entry][icl],
                                                           lastcell_clusters[entry][icl], nLayers)
        

            # Calculate and draw momentum and direction of primary particle
            # works with np, not tested with ak
            # assume it is the first particle in the array, for each event
            px = np.array([x[0] for x in px_part])
            py = np.array([x[0] for x in py_part])
            pz = np.array([x[0] for x in pz_part])
            p_part = np.sqrt(px**2 + py**2 + pz**2)
            # works with both np and ak
            # px = np.concatenate(px_part)
            # py = np.concatenate(py_part)
            # pz = np.concatenate(pz_part)
            # p_part = np.sqrt(np.concatenate(px_part)**2 + np.concatenate(py_part)**2 + np.concatenate(pz_part)**2)
            phi_part = np.arctan2(py, px)
            theta_part = np.arccos(pz/p_part)

            # plot generated spectra only for 1st cluster collection
            if clusters == clusterCollections[0]:
                # plot the generated particle spectra - energy
                E=energy/1000.
                plt.clf()
                plt.hist(p_part,100,(E-0.99,E+1.01))
                plt.xlabel('$E_{true}$ [GeV]')
                plt.savefig('plots/e_true_energy_%d.pdf' % energy)
                #plt.show()
        
                # plot the generated particle spectra - theta
                plt.clf()
                plt.hist(theta_part*180/np.pi,50)
                plt.xlabel('$\\theta_{true}$ $[{}^\\circ]$')
                plt.savefig('plots/theta_true_energy_%d.pdf' % energy)
                #plt.show()
        
                # plot the generated particle spectra - phi
                plt.clf()
                plt.hist(phi_part,50)
                plt.xlabel('$\phi_{true}$')
                plt.savefig('plots/phi_true_energy_%d.pdf' % energy)
                #plt.show()
    
            # Calculate and draw cluster distributions
        
            # calculate energy and position of clusters with max energy
            keep = np.ones(nentries, dtype=bool)
            e_cl = np.zeros(nentries)
            theta_cl = np.zeros(nentries)
            phi_cl = np.zeros(nentries)
            rho_cl = np.zeros(nentries)
            imax = np.array([-1]*nentries)
            ecl_vs_layer = np.zeros((nentries, nLayers))

            if useAK:
                # TODO: implement keep mask in case len(e_clusters[entry]==0)
                e_cl = ak.max(e_clusters, axis=1).to_numpy()
                e_cl_pos = ak.argmax(e_clusters,  axis=-1, keepdims=True, mask_identity=False)
                imax = ak.argmax(e_clusters, axis=1).to_numpy()
                x_cl = x_clusters[ak.from_regular(e_cl_pos)].to_numpy().flatten()
                y_cl = y_clusters[ak.from_regular(e_cl_pos)].to_numpy().flatten()
                z_cl = z_clusters[ak.from_regular(e_cl_pos)].to_numpy().flatten()
                rho_cl = np.sqrt(x_cl**2 + y_cl**2)
                phi_cl = np.arctan2(y_cl, x_cl)
                theta_cl = np.arctan2(rho_cl, z_cl)
            else:
                for entry in range(nentries):
                    if len(e_clusters[entry])==0:
                        print('No clusters found for event %d, p=%f, skipping' % (entry, p_part[entry]))
                        keep[entry] = 0
                    else:
                        imax[entry] = np.argmax(e_clusters[entry])
                        icl = imax[entry]
                        # e, theta, phi, rho of cluster
                        e_cl[entry] = e_clusters[entry][icl]
                        rho_cl[entry] = sqrt(x_clusters[entry][icl]**2 + y_clusters[entry][icl]**2)
                        phi_cl[entry] = atan2(y_clusters[entry][icl],x_clusters[entry][icl])
                        theta_cl[entry] = atan2(rho_cl[entry], z_clusters[entry][icl])
                        if useShapeParameters:
                            for layer in range(nLayers):
                                position = parBegin[entry][icl] + inputFeaturePositions[energy][clusters][layer]
                                ecl_vs_layer[entry, layer] = shapeParameters[entry][position]

            if not useShapeParameters:
                # Calculate and draw cluster energies per layer
                # calculate cluster energy for each layer
                ecl_vs_layer = np.zeros((nentries, nLayers))
                # this for loop could perhaps be rewritten leveraging numpy method..
                for entry in range(nentries):
                    # get index of most energetic cluster
                    icl = imax[entry]
                    # calculate energy of each layer
                    if useAK:
                        ecl_vs_layer[entry] = calc_energy_per_layer_ak(entry, icl)
                    else:
                        ecl_vs_layer[entry] = calc_energy_per_layer(entry, icl)
            
            if debug: print(ecl_vs_layer)

            # remove events without clusters
            p_part = p_part[keep]
            e_cl = e_cl[keep]
            rho_cl = rho_cl[keep]
            phi_cl = phi_cl[keep]
            theta_cl = theta_cl[keep]
            imax = imax[keep]
            ecl_vs_layer = ecl_vs_layer[keep]
            if useShapeParameters:
                energyFraction_vs_layer = ecl_vs_layer
            else:
                energyFraction_vs_layer = ecl_vs_layer/e_cl[:,None]

            nentries = len(p_part)
            print('Total number of events selected:', nentries)
            print('')

            # calculate additional features
            theta_cl_mod_calo = theta_cl/deltaThetaCalo % 1
            phi_cl_mod_calo = phi_cl/deltaPhiCalo % 1

            # plot the reconstructed cluster E, theta, phi
            plt.clf()
            plt.hist(e_cl,100,(E*0.8,E*1.2))
            plt.xlabel('$E_{cl}$ [GeV]')
            plt.savefig('plots/e_reco_energy_%d_%s.pdf' % (energy, clusters))
            #plt.show()

            plt.clf()
            plt.hist(theta_cl*180/np.pi,50)
            plt.xlabel('$\\theta_{cl}$ $[{}^\\circ]$')
            plt.savefig('plots/theta_reco_energy_%d_%s.pdf' % (energy, clusters))
            #plt.show()
        
            plt.clf()
            plt.hist(phi_cl,50)
            plt.xlabel('$\phi_{cl}$')
            plt.savefig('plots/phi_reco_energy_%d_%s.pdf' % (energy, clusters))
        
            #plt.close()
        
        
            # draw the cluster energy per layer
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            plt.clf()
            ax = plt.subplot(1,1,1)
            for layer in range(nLayers):
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                ax.clear()
                ax.text(0.1,0.85,'E = %.1f GeV' % (energy/1000.), transform=ax.transAxes)
                ax.text(0.1,0.8,'Layer %d' % layer, transform=ax.transAxes)

                ax.hist(ecl_vs_layer[:,layer], 50)
                ax.set_xlabel('Energy in layer [GeV]')
                # plt.legend()
                # plt.savefig('plots/energy_in_layer%d_%s.pdf' % (layer, clusters))
                # plt.show()

            # calculate and draw energy fraction per layer
            if debug:
                print("Cluster energy:")
                print(e_cl)
                print("Cluster energy fraction per layer:")
                print(energyFraction_vs_layer)

                for layer in range(nLayers):
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111)
                    ax.clear()
                    ax.text(0.1,0.8,'Layer %d' % layer, transform=ax.transAxes)
                    ax.text(0.1,0.75,'<f> = %f' % np.mean(energyFraction_vs_layer[:,layer]), transform=ax.transAxes)

                    ax.hist(energyFraction_vs_layer[:,layer], 50)
                    ax.set_xlabel('Energy in layer / Cluster energy')
                    # plt.legend()
                    # plt.savefig('plots/energy_in_layer%d_%s.pdf' % (layer, clusters))
                    # plt.show()
    
            # calculate the residuals
            de = e_cl - p_part
            de_over_e = de/p_part

            # plot the energy residuals (cluster-particle)
            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            ax.clear()
    
            ax.hist(de_over_e, bins=50, label='raw', alpha=0.5, range=(-0.3,0.3))
            mean = np.mean(de_over_e)
            std = np.std(de_over_e)
            meanerr = std/sqrt(len(de_over_e))
            stderr = std/sqrt(2*len(de_over_e)-2)
            ax.text(0.7,0.8,'raw:', transform=ax.transAxes)
            ax.text(0.7,0.75,'avg = %f +- %f' % (mean, meanerr), transform=ax.transAxes)
            ax.text(0.7,0.7,'std = %f +- %f' % (std, stderr), transform=ax.transAxes)

            ax.set_xlabel('$(E_{cl}^{raw}/E_{true}) - 1$')
            plt.legend()
            #plt.show()
            plt.savefig('plots/e_resol_raw_energy_%d_%s.pdf' % (energy, clusters))


            # Do the regression

            # pack the energy fractions and total energy into a numpy array for input to the regressor
            # and calculate the calibrated energy
            inputs = [energyFraction_vs_layer.transpose()]
            if useExtraFeatures: inputs.extend([theta_cl, theta_cl_mod_calo, phi_cl_mod_calo])
            inputs.append(e_cl)
            data = np.vstack(inputs)
            predict = []
            if modelFormat=='lgbm':
                datat = data.transpose()
                for i in range(ncalibrations):
                    predict.append(model[i].predict(datat))
            else:
                datat = data.transpose().astype(np.float32)
                for i in range(ncalibrations):
                    outputs = session[i].run(None, {"X" : datat})
                    predict.append(outputs[0].flatten())
            prediction = np.full_like(e_cl, np.nan, dtype=np.float32)
            i=0
            for calibration in calibrations:
                emin_cl = calibration["emin_cl"]
                emax_cl = calibration["emax_cl"]
                # print(emin_cl, emax_cl)
                prediction = np.where((e_cl>=emin_cl) & (e_cl<emax_cl), predict[i], prediction)
                i+=1
            e_calib = e_cl*prediction
            de_calib = e_calib - p_part
            de_calib_over_e = de_calib/p_part

            # compare raw and calibrated energies
            ax.clear()
            egev = energy/1000.
            res = sqrt(0.1*0.1/egev/egev + 0.02*0.02)
            xmax = 5*res
            xmin = -5*res
            nbins = 100
            plt.xlim(xmin, xmax)
            x = np.linspace(xmin, xmax, nbins+1)
            counts, bins, patches = ax.hist(de_over_e, bins=x, label='raw', alpha=0.5, range=(xmin,xmax))
            centers = (0.5*(bins[1:]+bins[:-1]))
            mean = np.mean(de_over_e)
            std = np.std(de_over_e)
            meanerr = std/sqrt(len(de_over_e))     # should be same as scipy.stats.sem(de_over_e)
            stderr = std/sqrt(2*len(de_over_e)-2)
            responses_raw[clusters][idx] = mean*100
            resolutions_raw[clusters][idx] = std*100/(1+mean)

            ax.text(0.07, 0.88, 'E = %.1f GeV' % (energy/1000.), transform=ax.transAxes)
            ax.text(0.07, 0.83, clusters, transform=ax.transAxes)
            ax.text(0.7, 0.80, 'raw:', transform=ax.transAxes)
            ax.text(0.7, 0.75, 'avg = %.3f $\pm$ %.3f' % (mean, meanerr), transform=ax.transAxes)
            ax.text(0.7, 0.70, 'std = %.3f $\pm$ %.3f' % (std, stderr), transform=ax.transAxes)
            print('Response bias raw from hist: %.1f%%' % (mean*100))
            print('Resolution raw from hist: %.1f%%' % (std*100))
        
            (_mu,_sigma) = norm.fit(de_over_e)
            a = len(de_over_e)*(xmax-xmin)/nbins
            pars, pcov = opt.curve_fit(lambda x, m, sig: gauss(x, a, m, sig), centers, counts, p0=(_mu, _sigma))
            perr = np.sqrt(np.diag(pcov))
            _mu = pars[0]
            _muerr = perr[0]
            _sigma = pars[1]
            _sigmaerr = perr[1]
            ax.plot(centers, a*norm.pdf(centers, _mu, _sigma))
            ax.text(0.7,0.65,'$\mu$ = %.3f $\pm$ %.3f' % (_mu, _muerr), transform=ax.transAxes)
            ax.text(0.7,0.60,'$\sigma$ = %.3f $\pm$ %.3f' % (_sigma, _sigmaerr), transform=ax.transAxes)
            ax.set_xlabel('$(E_{cl}^{raw}/E_{true}) - 1$')
            plt.legend()
            plt.savefig('plots/e_resol_raw_energy_%d_%s.pdf' % (energy, clusters))

            if useParamsFromFits:
                responses_raw[clusters][idx] = _mu*100
                resolutions_raw[clusters][idx] = _sigma*100./(1+_mu)
                #resolutions_raw_err[clusters][idx] = _sigmaerr*100./(1+_mu)
            print('Response bias raw from fit: %.1f%%' % (_mu*100))
            print('Resolution raw from fit: %.1f%%' % (_sigma*100/(1+_mu)))

            counts, bins, patches = ax.hist(de_calib_over_e, bins=x,
                                            label='calib', alpha=0.5, range=(xmin,xmax))
            centers = (0.5*(bins[1:]+bins[:-1]))
            mean = np.mean(de_calib_over_e)
            std = np.std(de_calib_over_e)
            meanerr = std/sqrt(len(de_calib_over_e))
            stderr = std/sqrt(2*len(de_calib_over_e)-2)
            responses_cal[clusters][idx] = mean*100.
            resolutions_cal[clusters][idx] = std*100./(1+mean)
            resolutions_cal_err[clusters][idx] = stderr*100.
            ax.text(0.7,0.50,'calib:', transform=ax.transAxes)
            ax.text(0.7,0.45,'avg = %.3f $\pm$ %.3f' % (mean, meanerr), transform=ax.transAxes)
            ax.text(0.7,0.40,'std = %.3f $\pm$ %.3f' % (std, stderr), transform=ax.transAxes)
            print('Response bias cal from hist: %.1f%%' % (mean*100))
            print('Resolution cal from hist: %.1f%%' % (std*100/(1.+mean)))

            (_mu,_sigma) = norm.fit(de_calib_over_e)
            a = len(de_calib_over_e)*(xmax-xmin)/nbins
            pars, pcov = opt.curve_fit(lambda x, m, sig: gauss(x, a, m, sig), centers, counts, p0=(_mu, _sigma))
            perr = np.sqrt(np.diag(pcov))
            _mu = pars[0]
            _muerr = perr[0]
            _sigma = pars[1]
            _sigmaerr = perr[1]
            ax.plot(centers, a*norm.pdf(centers, _mu, _sigma))
        
            ax.text(0.7,0.35,'$\mu$ = %.3f $\pm$ %.3f' % (_mu, _muerr), transform=ax.transAxes)
            ax.text(0.7,0.30,'$\sigma$ = %.3f $\pm$ %.3f' % (_sigma, _sigmaerr), transform=ax.transAxes)

            ax.set_xlabel('$(E_{cl}^{raw}/E_{true}) - 1$')
            plt.legend()
            plt.savefig('plots/e_resol_calib_energy_%d_%s.pdf' % (energy, clusters))

            if useParamsFromFits:
                responses_cal[clusters][idx] = _mu*100
                resolutions_cal[clusters][idx] = _sigma*100./(1+_mu)
                resolutions_cal_err[clusters][idx] = _sigmaerr*100./(1+_mu)
            print('Response bias cal from fit: %.1f%%' % (_mu*100))
            print('Resolution cal from fit: %.1f%%' % (_sigma*100/(1+_mu)))

            # TODO: understand why this sometimes fails...
            #with open("plots/e_resol_calib_energy_%d_%s.pkl" % (energy, clusters), "wb") as f:
            #    pickle.dump((plt.gcf(), ax), f)
            idx+=1

    # TODO: re-enable when previous bug in pickle.dump is fixed
    # attempt to draw all resolution plots in a single, multi-pad, figure
    #plt.clf()
    #plots = []
    #for i, energy in enumerate(energies):
    #    (fig, ax) = pickle.load(open("plots/e_resol_calib_energy_%d_%s.pkl" % (energy, clusters), "rb"))
    #    plots.append((fig,ax))
    #    plt.close()
    #    os.remove("plots/e_resol_calib_energy_%d_%s.pkl" % (energy, clusters))
        
    # Create a single multi-plot figure
    # num_rows = 4
    # num_cols = 3
    # f, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    # f.suptitle(clusters)
    # # Iterate through the list of plots and draw them on the subplots
    # for i, (fig,ax) in enumerate(plots):
    #     row = i // num_cols
    #     col = i % num_cols
    #     axs[row, col].axis('off')  #  Turn off axis for cleaner layout
    #     axs[row, col].figure.subplots_adjust(wspace=0, hspace=0)
    #     axs[row, col].imshow(fig2img(fig), aspect='auto', extent=axs[row, col].get_position().bounds)

    # for i in range(len(plots), num_rows*num_cols):
    #     row = i // num_cols
    #     col = i % num_cols
    #     axs[row,col].axis('off')

    # f.savefig('plots/e_resol_calib_%s.pdf' % clusters, dpi=600)


# save data to json so that we can redo the final plots later (maybe changing style)
# without redoing all the fits
dataForJson = {}
dataForJson["energies"] = energies
dataForJson["responses_raw"] = responses_raw
dataForJson["resolutions_raw"] = resolutions_raw
dataForJson["responses_cal"] = responses_cal
dataForJson["resolutions_cal"] = resolutions_cal
dataForJson["resolutions_cal_err"] = resolutions_cal_err
if not readDataFromJson:
    with open(label+'.json', 'w') as json_file:
        json.dump(dataForJson, json_file, indent=4)

# do the final plots of resolution and response vs energy for all
# cluster types
print('\nProducing the final plots...')
plt.clf()
for clusters in clusterCollections:
    plt.scatter(energies_in_gev, responses_raw[clusters], label='%s' % clusters)
    plt.scatter(energies_in_gev, responses_cal[clusters], label='Calib%s' % clusters)
plt.legend(loc='best', fontsize=14)
plt.grid()
plt.xlabel('$E_{true}$ [GeV]', fontsize=16)
plt.ylabel('Response bias [%]', fontsize=16)
plt.savefig(f'plots/response_vs_energy_{label}.pdf')

plt.clf()
sc = {}
for clusters in clusterCollections:
    plt.scatter(energies_in_gev, resolutions_raw[clusters], label='%s' % clusters)
    sc[clusters] = plt.scatter(energies_in_gev, resolutions_cal[clusters], label='Calib%s' % clusters)
plt.legend(loc='best', fontsize=14)
plt.grid()
plt.xlabel('$E_{true}$ [GeV]', fontsize=16)
plt.ylabel('Resolution [%]', fontsize=16)
plt.savefig(f'plots/resolution_vs_energy_{label}.pdf')

for clusters in clusterCollections:
    print('\nCluster collection = %s\n' % clusters)
    if doNoise:
        popts, pcov = opt.curve_fit(resol_curve, energies_in_gev, resolutions_cal[clusters], sigma=resolutions_cal_err[clusters], p0=(0.1, 7.7, 0.2))
    else:    
        popts, pcov = opt.curve_fit(resol_curve_no_noise, energies_in_gev, resolutions_cal[clusters], sigma=resolutions_cal_err[clusters], p0=(7.7, 0.2))
    #popts, pcov = opt.curve_fit(resol_curve_with_d, energies_in_gev, resolutions_cal[clusters], sigma=resolutions_cal_err[clusters], p0=(7, 0., 0.))
    perr = np.sqrt(np.diag(pcov))
    popts = np.abs(popts)
    fitparams[clusters] = popts
    
    if doNoise:
        print('Noise = %.0f +- %.0f MeV' % (popts[0]*10, perr[0]*10))
        print('a = %.1f +- %.1f %%' % (popts[1], perr[1]))
        print('c = %.1f +- %.1f %%' % (popts[2], perr[2]))
    else:
        print('a = %.1f +- %.1f %%' % (popts[0], perr[0]))
        print('c = %.1f +- %.1f %%' % (popts[1], perr[1]))

    col = sc[clusters].get_facecolors()[0].tolist()
    # linear interpolation
    # xvals_curve = np.linspace(min(energies_in_gev), max(energies_in_gev), 200)
    # log(E) interpolation - smoother curve at low E
    xvals_curve = np.exp(np.linspace(log(min(energies_in_gev)), log(max(energies_in_gev)), 200))
    if doNoise:
        plt.plot(xvals_curve,
                 resol_curve(xvals_curve, *popts),
                 linestyle='-',
                 linewidth=2.5,
                 color=col,
                 label="$\\frac{{{0:.2f}}}{{E}}\oplus \\frac{{{1:.1f}\\%}}{{\\sqrt{{E}}}}\\oplus {2:.1f}\\%$".format(popts[0]*0.01, popts[1], popts[2]))
    else:
        plt.plot(xvals_curve,
                 resol_curve_no_noise(xvals_curve, *popts),
                 linestyle='-',
                 linewidth=2.5,
                 color=col,
                 label="$\\frac{{{0:.1f}\\%}}{{\\sqrt{{E}}}}\\oplus {1:.1f}\\%$".format(popts[0], popts[1]))
        
plt.xlabel('$E_{true}$ [GeV]')
plt.ylabel('Resolution [%]')
plt.legend(loc='best')
plt.savefig(f'plots/resolution_vs_energy_fit_cal_{label}.pdf')
#plt.show()

plt.xscale('log')

plt.savefig(f'plots/resolution_vs_logenergy_fit_cal_{label}.pdf')
plt.xscale('linear')

# plot resolutions vs 1/sqrt(E)
plt.clf()
sc = {}
for clusters in clusterCollections:
    print('\nCluster collection = %s\n' % clusters)
    plt.scatter(invsqrtenergies, resolutions_raw[clusters], label='%s' % clusters)
    sc[clusters] = plt.scatter(invsqrtenergies, resolutions_cal[clusters], label='Calib%s' % clusters)
    if doNoise:
        popts, pcov = opt.curve_fit(resol_curve_vs_invsqrtE, invsqrtenergies, resolutions_cal[clusters], sigma=resolutions_cal_err[clusters], p0=(0.07, 10, 1))
    else:
        popts, pcov = opt.curve_fit(resol_curve_vs_invsqrtE_no_noise, invsqrtenergies, resolutions_cal[clusters], sigma=resolutions_cal_err[clusters], p0=(10, 1))
    perr = np.sqrt(np.diag(pcov))
    popts = np.abs(popts)
    
    if doNoise:
        print('Noise = %.0f +- %.0f MeV' % (popts[0]*10, perr[0]*10))
        print('a = %.1f +- %.1f %%' % (popts[1], perr[1]))
        print('c = %.1f +- %.1f %%' % (popts[2], perr[2]))
    else:
        print('a = %.1f +- %.1f %%' % (popts[0], perr[0]))
        print('c = %.1f +- %.1f %%' % (popts[1], perr[1]))
    
    col = sc[clusters].get_facecolors()[0].tolist()
    #xvals_curve = np.linspace(energies_in_gev.min(), energies_in_gev.max(), 200)
    xvals_curve = np.linspace(min(invsqrtenergies), max(invsqrtenergies), 200)
    popts = np.abs(popts)
    if doNoise:
        plt.plot(xvals_curve,
                 resol_curve_vs_invsqrtE(xvals_curve, *popts),
                 linestyle='-',
                 linewidth=2.5,
                 color=col,
                 label="$\\frac{{{0:.2f}}}{{E}}\oplus \\frac{{{1:.1f}\\%}}{{\\sqrt{{E}}}}\\oplus {2:.1f}\\%$".format(popts[0]*0.01, popts[1], popts[2]))
    else:
        plt.plot(xvals_curve,
                 resol_curve_vs_invsqrtE_no_noise(xvals_curve, *popts),
                 linestyle='-',
                 linewidth=2.5,
                 color=col,
                 label="$\\frac{{{0:.1f}\\%}}{{\\sqrt{{E}}}}\\oplus {1:.1f}\\%$".format(popts[0], popts[1]))
        
plt.xlabel('$1/\sqrt{E_{true}}$ [GeV$^{-1/2}$]')
plt.ylabel('Resolution [%]')
plt.legend(loc='best')
plt.savefig(f'plots/resolution_vs_invsqrtenergy_fit_cal_{label}.pdf')
#plt.show()

# plot calibrated resolution and response on same plot
for clusters in clusterCollections:
    popts = fitparams[clusters]
    plt.clf()
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('$E_{true}$ [GeV]', fontsize=16)
    ax1.set_ylabel('Resolution [%]', fontsize=16)
    ax1.spines["left"].set_color(color)
    ax1.yaxis.label.set_color(color)
    ax1.tick_params(axis="y", colors=color)
    xvals_curve = np.exp(np.linspace(log(min(energies_in_gev)), log(max(energies_in_gev)), 200))
    line1 = ax1.scatter(energies_in_gev, resolutions_cal[clusters], label='Resolution', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    if doNoise:
        line2, = ax1.plot(xvals_curve,
                 resol_curve(xvals_curve, *popts),
                 linestyle='-',
                 linewidth=2.5,
                 color=color,
                 label="$\\frac{{{0:.2f}}}{{E}}\oplus \\frac{{{1:.1f}\\%}}{{\\sqrt{{E}}}}\\oplus {2:.1f}\\%$".format(popts[0]*0.01, popts[1], popts[2]))
    else:
        line2, = ax1.plot(xvals_curve,
                 resol_curve_no_noise(xvals_curve, *popts),
                 linestyle='-',
                 linewidth=2.5,
                 color=color,
                 label="$\\frac{{{0:.1f}\\%}}{{\\sqrt{{E}}}}\\oplus {1:.1f}\\%$".format(popts[0], popts[1]))
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Response bias [%]', color=color, fontsize=16)  # we already handled the x-label with ax1
    line3 = ax2.scatter(energies_in_gev, responses_cal[clusters], label='Response bias', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.axhline(y=0, color=color, linestyle='--', linewidth=1)
    ax2.set_ylim(-2.2,2.2);
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    # ax1.legend(lines, labels, title='Calib%s' % clusters, loc='best', frameon=False)
    title = ""
    if clusters.startswith("EMBCaloClusters"):
        title = "EM SW clusters"
    elif clusters.startswith("EMBCaloTopoClusters"):
        title = "EM topo clusters"
    if clusters.endswith("WithNoise"):
        title += ", noise on"
    else:
        title += ", noise off"
    # ax1.legend(lines, labels, title='Calib%s' % clusters, loc='best', frameon=False)
    ax1.legend(lines, labels, title=title, loc='best', frameon=False, fontsize=14)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'plots/resolution_and_response_vs_energy_cal_{clusters}.pdf')

dataForJson["fitparams"] = {}
for clusters in clusterCollections:
    dataForJson["fitparams"][clusters] = list(fitparams[clusters])
with open(label+'_with_fitparams.json', 'w') as json_file:
    json.dump(dataForJson, json_file, indent=4)
