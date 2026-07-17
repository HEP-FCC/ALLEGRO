#!/usr/bin/bash

# script to run the full chain
# from LAr_scripts with several additions (via FCC-scripts)
#
# run with ./run_all_chain.sh
#
# WIP some things still need to be ported to ALLEGRO from FCC-scripts
#
# Note: if the geometry is standard, some steps might not be needed, and the default files for e.g. noise, neighbours, ... could be used

runname="paper_LArPb"
xmlbasedir="${K4GEO:-../k4geo}"
xmldir=FCCee/ALLEGRO/compact/ALLEGRO_o1_v03
xmlfileFullDet=ALLEGRO_o1_v03
# base name of xml file for ECal barrel - should be the same in ALLEGRO_o1_v03.xml and in runParallel.py
xmlfileECal=ECalBarrel_thetamodulemerged
today=`date +%y%m%d`

# create GDML for event display
doGDML=0
# create xml files needed for sampling and upstream/downstream correction studies
doCalibrationFiles=0
# create plots of X/XO
doX0plots=0
# create maps of cell neighbours for topoclustering
doNeighbourMaps=0
# calculate sampling fractions
doSamplingFractions=0
# study SF vs E, theta
doStudySFvsETheta=0
# create map of cell noise for topoclustering and noise tools
# note: needs sampling fractions
doNoiseMaps=0
# calculate up/downstream corrections
doUpDownStreamCorrections=0
# generate events with up/downstream corrections applied to study resolution post-correction
# doClustersWithUpDownStreamCorrections=0
# produce electron and photon samples for training the MVA energy calibration
doClustersForMVATraining=0
# run the energy MVA calibration with XGBoost
# currently broken (problems with RDataFrame)
# use LGBM training instead (in ALLEGRO/mva_cluster_calibrations, see README there)
doMVATraining=0
# produce electron and photon samples for testing the MVA energy calibration
doClustersForMVAEvaluation=0
# calculate resolutions
doComputeResolutions=0
# produce clusters adding also output of MVA calibration and photonID tool
doClustersWithCalibrations=0



doSimulation=0
doReconstruction=0
doProductionSim=0
doProductionReco=0

mkdir -p log plots
mkdir -p $runname/log
mkdir -p $runname/plots
mkdir -p $runname

exec_cmd() {
    local cmd="$1"
    local log="$2"
    local mode="${3:-overwrite}"   # overwrite (default) or append
    echo
    echo "Executing command $cmd"
    echo "Output will be recorded in file $log"
    echo
    # this will also show the output to screen
    # $cmd 2>&1 | tee $log
    # this redirects the output to file
    if [ "$mode" = "append" ]; then
        eval "$cmd" >>"$log" 2>&1
    else
        eval "$cmd" >"$log" 2>&1
    fi
    retcode=${PIPESTATUS[0]}
    if [ $retcode -eq 0 ]; then
	echo -e "\e[1;42mDONE\e[0m"
	echo
    else
	echo -e "\e[1;41mFAILED\e[0m"
	exit
    fi
}

# Create the GDML detector model
#
if (( $doGDML > 0 )); then
    echo "Creating the GDML file of the detector..."
    exec_cmd "ddsim --numberOfEvents 0 --enableGun --compactFile $K4GEO/$xmldir/$xmlfileFullDet.xml --geometry.dumpGDML $xmlfileFullDet.gdml --runType run" "$runname/log/createGDML.log"
fi


# Remake calibration xml files from the main xml file
# [Note] usually not needed since the calibration files are already in k4geo,
#        and should be identical to the ones produced here. Do this if the main calo xml
#        has been modified, so that you propagate the changes consistently to the calibration files
#
if (( $doCalibrationFiles > 0 )); then
    echo "Creating the calibration files..."
    exec_cmd "python -u ../ALLEGRO/sampling_fractions/write_calibration_xml.py $xmlbasedir/$xmldir/$xmlfileECal.xml" "$runname/log/createCalibrationFiles.log"
fi


# Compute the X0 plot (material upstream and ECAL separately, and then full detector)
#
if (( $doX0plots > 0 )); then

    tracker=1
    ecal=1
    fulldet=1

    mkdir -vp $runname/geometry
    mkdir -vp $runname/geometry/xml
    mkdir -vp $runname/geometry/plots
    mkdir -vp $runname/geometry/python
    mkdir -vp $runname/geometry/root
    mkdir -vp $runname/geometry/log

    # 1. tracker only
    if (( $tracker > 0 )); then
	# - prepare steering file
	cp -f ../ALLEGRO/geometry/material_scan.py material_scan_tracker.py
	sed -i 's%#suffix%tracker%' material_scan_tracker.py
	sed -i 's%#etamax%2.7%' material_scan_tracker.py
	sed -i 's%#etabinning%0.1%' material_scan_tracker.py
	# - scan
	exec_cmd "k4run material_scan_tracker.py --trackeronly" "$runname/geometry/log/material_scan_tracker.log"
	# - plot vs costheta
	exec_cmd "python -u ../ALLEGRO/geometry/material_plot_vs_theta.py --f out_material_scan_tracker.root --s _tracker -c 1.0" "$runname/geometry/log/material_plot_tracker_vs_costheta.log"
	# - plot vs theta
	exec_cmd "python -u ../ALLEGRO/geometry/material_plot_vs_theta.py --f out_material_scan_tracker.root --s _tracker -t 0.0" "$runname/geometry/log/material_plot_tracker_vs_theta.log"
	mv material_scan_tracker.py $runname/geometry/python
	mv out_material_scan_tracker.root $runname/geometry/root
    fi

    # 2. ecal only
    if (( $ecal > 0 )); then
	# - prepare steering file
	cp -f ../ALLEGRO/geometry/material_scan.py material_scan_ecal.py
	sed -i 's%#suffix%ecal%' material_scan_ecal.py
	sed -i 's%#etamax%2.9%' material_scan_ecal.py
	sed -i 's%#etabinning%0.1%' material_scan_ecal.py
	# - scan
	exec_cmd "k4run material_scan_ecal.py --ecalonly --ecalbarrelXML ${xmlfileECal}.xml" "$runname/geometry/log/material_scan_ecal.log"
	# - plot vs costheta
	exec_cmd "python -u ../ALLEGRO/geometry/material_plot_vs_theta.py --f out_material_scan_ecal.root --s _ecal -c 1.0" "$runname/geometry/log/material_plot_ecal_vs_costheta.log"
	# - plot vs theta
	exec_cmd "python -u ../ALLEGRO/geometry/material_plot_vs_theta.py --f out_material_scan_ecal.root --s _ecal -t 0.0" "$runname/geometry/log/material_plot_ecal_vs_theta.log"
	mv material_scan_ecal.py $runname/geometry/python
	mv out_material_scan_ecal.root $runname/geometry/root
    fi

    # 3. full detector
    if (( $fulldet > 0 )); then
	# - prepare steering file
	cp -f ../ALLEGRO/geometry/material_scan.py material_scan_all.py
	sed -i 's%#suffix%all%' material_scan_all.py
	sed -i 's%#etamax%2.7%' material_scan_all.py
	sed -i 's%#etabinning%0.1%' material_scan_all.py
	# - scan
	exec_cmd "k4run material_scan_all.py" "$runname/geometry/log/material_scan_all.log"
	# - plot vs costheta
	exec_cmd "python -u ../ALLEGRO/geometry/material_plot_vs_theta.py --f out_material_scan_all.root --s _all -c 1.0" "$runname/geometry/log/material_plot_all_vs_costheta.log"
	# - plot vs theta
	exec_cmd "python -u ../ALLEGRO/geometry/material_plot_vs_theta.py --f out_material_scan_all.root --s _all -t 0.0" "$runname/geometry/log/material_plot_all_vs_theta.log"
	mv material_scan_all.py $runname/geometry/python
	mv out_material_scan_all.root $runname/geometry/root
    fi

    # Archive the files
    #
    cp $xmlbasedir/$xmldir/*.xml $runname/geometry/xml
    mv plots/material/*png $runname/geometry/plots
    mv plots/material/*pdf $runname/geometry/plots
fi


# Create neighbour maps for topoclustering
#
if (( $doNeighbourMaps > 0 )); then
    mkdir -p $runname/data
    echo "Creating the ecal neighbour map..."
    exec_cmd "k4run ../ALLEGRO/neighbor_maps/neighbours.py --detector $xmldir/$xmlfileFullDet.xml --ecalb --ecalec --link-ecal" "$runname/log/createNeighboursEcal.log"
    exec_cmd "k4run ../ALLEGRO/neighbor_maps/neighbours.py --detector $xmldir/$xmlfileFullDet.xml --ecalb" "$runname/log/createNeighboursEcalB.log"
    exec_cmd "k4run ../ALLEGRO/neighbor_maps/neighbours.py --detector $xmldir/$xmlfileFullDet.xml --ecalec" "$runname/log/createNeighboursEcalE.log"
    mv neighbours_map_ecalB_ecalE.root $runname/data
    mv neighbours_map_ecalB.root $runname/data
    mv neighbours_map_ecalE.root $runname/data
    ln -s neighbours_map_ecalB.root $runname/data/neighbours_map_ecalB_thetamodulemerged.root
    ln -s neighbours_map_ecalE.root $runname/data/neighbours_map_ecalE_turbine.root
    echo "Creating the ecal+hcal neighbour map"
    exec_cmd "k4run ../ALLEGRO/neighbor_maps/neighbours.py --detector $xmldir/$xmlfileFullDet.xml --ecalb --hcalb --ecalec --hcalec --link-calos --link-ecal --link-hcal" "$runname/log/createNeighboursEcalHcal.log"
    mv neighbours_map_ecalB_ecalE_hcalB_hcalE.root $runname/neighbours
fi


# Compute sampling fractions and update scripts
#
if (( $doSamplingFractions > 0 )); then
    echo "Computing sampling fractions"
    # - one energy is enough as they are independent of energy and direction
    # e-, 10 GeV, 90 degrees
    # python runParallel.py --outDir $runname/sampling --nEvt 10000 --energies 10000 --particles e- --sampling
    # e-, 1-10-20-50 GeV, 90 degrees
    # python runParallel.py --outDir $runname/sampling --nEvt 10000 --energies 1000 10000 20000 50000 --particles e- --sampling
    # e-, 20 GeV, flat theta spectrum (DEFAULT)
    exec_cmd "python -u runParallel.py --outDir $runname/sampling --nEvt 10000 --energies 20000 --thetas -1 --particles e- --sampling" "$runname/log/createSamplingCorrections.log"
    # x-check Jiashun numbers:
    # exec_cmd "python -u runParallel.py --outDir $runname/sampling --nEvt 10000 --energies 50000 --thetas -1 --particles gamma --sampling" "$runname/log/createSamplingCorrections.log"
    # gamma, 10 GeV, flat theta spectrum (to x-check particle dependence)
    # note that for low-energy photons the estimated sampling fraction in the first layers is lower than for electrons as the
    # shower starts later.
    # python runParallel.py --outDir $runname/sampling_photon --nEvt 10000 --energies 10000 --thetas -1 --particles gamma --sampling
fi

# - otherwise, to plot sampling fractions vs energy or direction and check directly this independence, one can do
if (( $doStudySFvsETheta > 0 )); then
    exec_cmd "python -u runParallel.py --outDir $runname/sampling --nEvt 2000 --energies 10000 20000 50000 100000 --sampling" "$runname/log/samplingVsE.log"
    exec_cmd "python -u runParallel.py --outDir $runname/sampling --nEvt 2000 --energies 10000 --thetas 90 80 70 60 50 --sampling" "$runname/log/samplingVsTheta.log"
    exec_cmd "cd ../ALLEGRO/sampling_fractions/FCC_calo_analysis_cpp" /dev/null
    exec_cmd "python plot_samplingFraction.py ../../../run/$runname/sampling/calibration_sampling_output_energy_?_theta_90.0_particle_e-.root 10 20 50 100 -r 10000 20000 50000 100000 --totalNumLayers 11 --preview -outputfolder plots_sampling_fraction_$today --plotSFvsEnergy --noFits" "../../../run/$runname/log/samplingVsE.log" "append"
    exec_cmd "python plot_samplingFraction.py ../../../run/$runname/sampling/calibration_sampling_output_energy_10000_theta_?_particle_e-.root 50 60 70 80 90 -r 50.0 60.0 70.0 80.0 90.0 --totalNumLayers 11 --preview -outputfolder plots_sampling_fraction_$today --plotSFvsEnergy --theta --noFits" "../../../run/$runname/log/samplingVsTheta.log" "append"
    echo "Plots of sampling fractions are produced in ../ALLEGRO/sampling_fractions/FCC_calo_analysis_cpp/plots_sampling_fraction_$today"
    exec_cmd "cd .." /dev/null
fi


# Create noise maps for digitisation and topoclustering
# Some scripts need the sampling fractions, so this step should be run after the sampling fraction calculation
#
if (( $doNoiseMaps > 0 )); then

    mkdir -p $runname/data
    exec_cmd "cd $runname/data" /dev/null

    # GM TODO REPLACE WITH ALLEGRO SCRIPTS
    # echo "Performing a dry run of the simulation (0 events) to obtain geometry parameters for the capacitance calculation..."
    # exec_cmd "ddsim --numberOfEvents 0 --enableGun --compactFile $K4GEO/$xmldir/$xmlfileFullDet.xml --runType run" "log/dryRunForNoise.log"
    #echo "Extracting geometry and segmentation parameters for the capacitance calculation..."
    #exec_cmd "python -u ../FCC-scripts/getECalBarrelNumbersForNoise.py" "log/getECalBarrelNumbersForNoise.log"
    #echo "Creating the ecal barrel capacitance histograms..."
    #exec_cmd "python -u ../FCC-scripts/create_capacitance_file_theta.py" "log/createCapacitanceECalBarrel.log"
    #echo "Creating the ecal barrel noise histograms..."
    #exec_cmd "python -u ../FCC-scripts/create_noise_file_chargePreAmp_theta.py" "log/createNoiseHistsECalBarrel.log"

    exec_cmd "python -u ../../../ALLEGRO/noise_maps/create_capacitance_file_theta_update2025.py" "../../log/createNoiseMaps.log"
    exec_cmd "python -u ../../../ALLEGRO/noise_maps/create_noise_file_chargePreAmp_theta_update2025.py" "../../log/createNoiseMaps.log" "append"
    exec_cmd "python -u ../../../ALLEGRO/noise_maps/create_capacitance_file_ecalendcap_simple.py $xmldir/$xmlfileFullDet.xml" "../../log/createNoiseMaps.log" "append"
    exec_cmd "python -u ../../../ALLEGRO/noise_maps/create_noise_file_ecalendcap.py $xmldir/$xmlfileFullDet.xml" "../../log/createNoiseMaps.log" "append"

    echo "Creating the ecal noise map..."
    exec_cmd "k4run ../../../ALLEGRO/noise_maps/noise_map.py --detector $xmldir/$xmlfileFullDet.xml --subdetectors ecalb ecale" "../../log/createNoiseMap.log" "append"

    echo "Creating the ecal+hcal noise map..."
    exec_cmd "k4run ../../../ALLEGRO/noise_maps/noise_map.py --detector $xmldir/$xmlfileFullDet.xml --subdetectors ecalb ecale hcalb hcale" "../../log/createNoiseMap.log" "append"

    echo "Noise maps created. To compare to a previous map to check if they are identical or not, you can use the compareMaps script"
    cd ../..
fi


# Compute upstream and downstream corrections and update scripts
# Note: the reconstruction jobs will occasionally throw warning messages that
# there is more than one input particle. in that case, the first one (corresponding
# to the generated electron) is used
if (( $doUpDownStreamCorrections > 0 )); then
    echo "Computing upstream/downstream corrections"
    # The script generates samples of particles of various energies, that are used to
    # calculate the profiles of E(upstream) vs E(layer 0) and of E(downstream) vs E(layer -1)
    # which are then fitted with some parametric functions
    # The values of the parameters vs particle energy are then fitted to obtain
    # a parameterisation of the corrections vs energy
    # flat theta between 85 and 95 degrees
    exec_cmd "python -u runParallel.py --outDir $runname/upstream --nEvt 5000 --energies 1000 5000 10000 15000 20000 30000 50000 75000 100000 --thetas -1 --particles e- --upstream --SF $runname/sampling/SF.json" "$runname/log/createUpDownStreamCorrections.log"
fi


# Generate clusters for upstream studies (can use --clusters instead)
#if (( $doClustersWithUpDownStreamCorrections > 0 )); then
#    echo "Producing clusters with upstream/downstream corrections"
#    exec_cmd "python -u runParallel.py --outDir $runname/upstreamProd --nEvt 1000000 --upstreamProd --SF $runname/sampling/SF.json" "log/createClustersWithUpDownStreamCorrections.log"
#fi


# Generate clusters for MVA training
if (( $doClustersForMVATraining > 0 )); then
    # Only 300k events here. Move up to 3M if needed
    # (~1M/day on APC server)

    # first, generate and simulate events with ddsim
    # python -u runParallel.py --outDir $runname/training_simulation --nEvt 300000 --particles gamma pi0 e- --simprod --thetaMinMax 40 140
    # python -u runParallel.py --outDir $runname/training_simulation_lowE --nEvt 10000 --particles gamma pi0 e- --simprod --thetaMinMax 40 140
    exec_cmd "python -u runParallel.py --outDir $runname/training_simulation_big2 --nEvt 300000 --particles gamma --simprod --thetaMinMax 40 140" "$runname/log/simulateEventsForMVATraining_big.log"
    # exec_cmd "python -u runParallel.py --outDir $runname/training_simulation_small_highE --nEvt 10000 --particles gamma --simprod --thetaMinMax 40 140" "$runname/log/simulateEventsForMVATraining_small_highE.log"

    # then, reconstruct them
    # - example to read SFs and corrections from external files
    # python runParallel.py --outDir $runname/production --nEvt 300000 --production --SF $runname/sampling/SF.json --corrections $runname/upstream/corr_params_1d.json

    # - example where total number of events is given. Job will be splitted in many subjobs, each running over the merged root file, each reconstructing a given number of events
    # exec_cmd "python -u runParallel.py --outDir $runname/training_reconstruction --nEvt 300000 --particles gamma pi0 e- --recprod --simInput $runname/training_simulation" "$runname/log/reconstructEventsForMVATraining.log"
    # small prod (10k)
    # exec_cmd "python -u runParallel.py --outDir $runname/training_reconstruction_small --nEvt 10000 --particles gamma e- --recprod --simInput $runname/training_simulation_small" "$runname/log/reconstructEventsForMVATraining_small.log"
    # exec_cmd "python -u runParallel.py --outDir $runname/training_reconstruction_small_highE --nEvt 10000 --particles gamma --recprod --simInput $runname/training_simulation_small_highE" "$runname/log/reconstructEventsForMVATraining_small_highE.log"
    # exec_cmd "python -u runParallel.py --outDir $runname/training_reconstruction_big --nEvt 300000 --particles gamma --recprod --simInput $runname/training_simulation_big" "$runname/log/reconstructEventsForMVATraining_big.log"
    exec_cmd "python -u runParallel.py --outDir $runname/training_reconstruction_big2 --nEvt 300000 --particles gamma --recprod --simInput $runname/training_simulation_big2" "$runname/log/reconstructEventsForMVATraining_big2.log"

    # - example where the number of input files per job is given, in that case the jobs will run over the unmerged root files
    # python -u runParallel.py --outDir $runname/training_reconstruction --inputFilesPerJob 5 --nEvt 300000 --particles gamma pi0 --recprod --simInput $runname/training_simulation
    # python -u runParallel.py --outDir $runname/training_reconstruction --inputFilesPerJob 5 --nEvt 10000 --particles gamma e- --recprod --simInput $runname/training_simulation

    # python -u runParallel.py --outDir $runname/training_reconstruction_smallSWclusters_noise_xtalk --nEvt 300000 --inputFilesPerJob 2 --particles gamma --recprod --simInput $runname/training_simulation --addNoise --addCrosstalk
    # python -u runParallel.py --outDir $runname/training_reconstruction_smallSWclusters_lowE_noise_xtalk --nEvt 10000 --particles gamma --recprod --simInput $runname/training_simulation_lowE --addNoise --addCrosstalk
    # python -u runParallel.py --outDir $runname/training_reconstruction_smallSWclusters_highE_noise_xtalk --nEvt 15000 --particles gamma --recprod --simInput $runname/training_simulation_highE --addNoise --addCrosstalk

    # - reconstruct low-E sample
    # python -u runParallel.py --outDir $runname/training_reconstruction_lowE --nEvt 10000 --particles gamma pi0 e- --recprod --simInput $runname/training_simulation_lowE
    # produce clusters with noise
    # python -u runParallel.py --outDir $runname/training_reconstruction_withnoise --nEvt 300000 --inputFilesPerJob 2 --particles gamma --recprod --simInput $runname/training_simulation --addNoise


    # for endcap
    # exec_cmd "python -u runParallel.py --outDir $runname/training_simulation_endcap_200k --nEvt 200000 --particles gamma --simprod --thetaMinMax 5 40" "$runname/log/simulateEventsForMVATrainingEndcap.log"
    # exec_cmd "python -u runParallel.py --outDir $runname/training_reconstruction_endcap_200k_topo --nEvt 200000 --particles gamma --recprod --simInput $runname/training_simulation_endcap_200k" "$runname/log/reconstructEventsForMVATrainingEndcap.log"
    # exec_cmd "python -u runParallel.py --outDir $runname/training_simulation_endcap_300k_0_22 --nEvt 300000 --particles gamma --simprod --thetaMinMax 5 40" "$runname/log/simulateEventsForMVATrainingEndcap.log"
    # exec_cmd "python -u runParallel.py --outDir $runname/training_reconstruction_endcap_300k_0_22 --nEvt 300000 --particles gamma --recprod --simInput $runname/training_simulation_endcap_300k_0_22" "$runname/log/reconstructEventsForMVATrainingEndcap.log"
fi


# Train the MVA on CaloClusters and CaloTopoClusters with XGBoost
if (( $doMVATraining > 0 )); then
    echo "MVA training with XGBoost is broken (problem with RDataFrame). Use https://gitlab.cern.ch/gmarchio/fcc-lar-energy-calibration instead"

    # python -u ../FCC-scripts/training.py EMBCaloClusters -p e- -i $runname/training_reconstruction/ -o $runname/training_emb_calo_electrons.json --useShapeParameters
    # python -u ../FCC-scripts/training.py CaloTopoClusters -i $runname/training_reconstruction/ -o $runname/training_topo.json
    # This instead will not run the training, just write numpy arrays with input features and target, to use a different MVA tool
    # python -u ../FCC-scripts/training.py EMBCaloClusters -i $runname/training_reconstruction/ -p e- -o $runname/training_emb_calo_electrons.json --no-training --writeFeatures $runname/training_reconstruction/features --writeTarget $runname/training_reconstruction/target --useShapeParameters
    # python -u ../FCC-scripts/training.py CaloTopoClusters -i $runname/production/ -o $runname/training_topo.json --no-training --writeFeatures $runname/production/features --writeTarget $runname/production/target
fi


# Produce events at various fixed energies and run clustering algs to form clusters to study resolutions
if (( $doClustersForMVAEvaluation > 0 )); then
    # without updated SFs and up/downstream corrections
    # exec_cmd "python -u runParallel.py --outDir $runname/clusters --nEvt 5000 --particles gamma e- --energies 300 500 1000 5000 10000 15000 20000 30000 50000 75000 100000 --clusters" "$runname/log/createClustersForEvaluation.log"
    # exec_cmd "python -u runParallel.py --outDir $runname/clusters --nEvt 5000 --particles gamma --energies 135000 180000 --clusters" "$runname/log/createClustersForEvaluation_highE.log"
    exec_cmd "python -u runParallel.py --outDir $runname/clusters --nEvt 5000 --particles gamma --energies 300 500 1000 5000 10000 15000 20000 30000 50000 75000 100000 --clusters --thetas 20" "$runname/log/createClustersForEvaluationEndcap.log"

    # sim done, but reco failed, so resubmitting only reco
    #exec_cmd "python -u runParallel.py --outDir $runname/clusters --nEvt 5000 --particles gamma e- --energies 300 500 1000 5000 10000 15000 20000 30000 50000 75000 100000 --clusters --simInput $runname/clusters" "$runname/log/createClustersForEvaluation.log"
    # exec_cmd "python -u runParallel.py --outDir $runname/clusters --nEvt 5000 --particles gamma --energies 135000 180000 --clusters --simInput $runname/clusters" "$runname/log/createClustersForEvaluation_highE.log"

    # with updated SFs and up/downstream corrections
    # exec_cmd "python -u runParallel.py --outDir $runname/clusters --nEvt 5000 --energies 300 500 1000 5000 10000 15000 20000 30000 50000 75000 100000 --clusters --SF $runname/sampling/SF.json --corrections $runname/upstream/corr_params_1d.json" "log/createClustersForEvaluation.log"
    # do clusters using same simulation of previous production but adding xtalk in reconstruction
    # exec_cmd "python -u runParallel.py --outDir $runname/clusters_with_xtalk --nEvt 5000 --particles gamma --energies 500 1000 5000 10000 15000 20000 30000 50000 75000 100000 --clusters --addCrosstalk --simInput $runname/clusters" "log/createClustersForEvaluationWithXTalk.log"
    # do clusters using same simulation of previous production but adding noise in reconstruction
    # exec_cmd "python -u runParallel.py --outDir $runname/clusters_with_noise --nEvt 5000 --particles gamma --energies 300 500 1000 5000 10000 15000 20000 30000 50000 75000 100000 --clusters --addNoise --simInput $runname/clusters" "log/createClustersForEvaluationWithNoise.log"
    # exec_cmd "python -u runParallel.py --outDir $runname/clusters_nothreshold_topo --nEvt 5000 --particles gamma --energies 300 500 1000 5000 10000 --clusters --simInput $runname/clusters" "log/createClustersWithoutThresholdTopo.log"
    # do clusters using same simulation of previous production but adding noise and xtalk in reconstruction
    # exec_cmd "python -u runParallel.py --outDir $runname/clusters_with_noise_and_xtalk --nEvt 5000 --particles gamma --energies 300 500 1000 5000 10000 15000 20000 30000 50000 75000 100000 --clusters --addNoise --addCrosstalk --simInput $runname/clusters" "log/createClustersForEvaluationWithNoiseAndXTalk.log"

    # python -u runParallel.py --outDir $runname/clusters_smallSWcluster_xtalk --nEvt 5000 --particles gamma --energies 300 500 1000 5000 10000 15000 20000 30000 50000 75000 100000 --clusters --addNoise --addCrosstalk --simInput $runname/clusters

    # python -u runParallel.py --outDir $runname/clusters_stdsize_noise_xtalk --nEvt 5000 --particles gamma --energies 300 500 1000 5000 10000 15000 20000 30000 50000 75000 100000 --clusters --addNoise --addCrosstalk --simInput $runname/clusters
    # python -u runParallel.py --outDir $runname/clusters_stdsize_noise_xtalk --nEvt 5000 --particles gamma pi0 --energies 300 500 1000 15000 20000 30000 50000 75000 100000 --clusters --addNoise --addCrosstalk --runPhotonID --calibrateClusters

    # noise debug: neutrinos + noise
    # exec_cmd "python -u runParallel.py --outDir $runname/clusters_with_noise --nEvt 50 --particles nu_e --energies 100000 --clusters --addNoise" "log/createClustersForEvaluationWithNoise.log"
fi

if (( $doClustersWithCalibrations > 0)); then
    python -u runParallel.py --outDir $runname/clusters_smallSWcluster --nEvt 5000 --particles gamma --energies 300 500 1000 5000 10000 15000 20000 30000 50000 75000 100000 --clusters --simInput $runname/clusters --calibrateClusters --runPhotonID
fi

# Compute resolutions and responses of the clusters produced in the previous step, also applying the MVA calibrations
if (( $doComputeResolutions > 0 )); then
    python compute_resolutions.py --inputDir $runname/clusters --outFile $runname/results.csv --clusters CaloClusters CorrectedCaloClusters CaloTopoClusters CorrectedCaloTopoClusters --MVAcalibCalo $runname/training_calo.json --MVAcalibTopo $runname/training_topo.json

    # Make resolution plots
    # - for each energy point estimate the responses and resolutions
    python plot_resolutions.py --outDir $runname --doFits plot $runname/results.csv --all
    # - compare the resolutions among different cluster collections and calibrations
    # 1. showing also raw clusters and clusters with up/downstream corrections
    python plot_resolutions.py --outDir $runname --doFits compare clusters CaloClusters CorrectedCaloClusters CaloTopoClusters CorrectedCaloTopoClusters CalibratedCaloClusters CalibratedCaloTopoClusters $runname/results.csv --all
    # 2. showing only the calibrated clusters
    python plot_resolutions.py --outDir $runname --doFits compare clusters CalibratedCaloClusters CalibratedCaloTopoClusters $runname/results.csv --all
fi


# Generic simulation and reconstruction jobs
if (( $doSimulation > 0 )); then
    # gamma and pi0, 20 GeV, 90 degrees, 1k events
    # python runParallel.py --outDir $runname/simulation --nEvt 1000 --energies 20000 --particles gamma pi0 --sim
    # e-, 10 GeV, 90 degrees, 10k events
    # python runParallel.py --outDir $runname/simulation --nEvt 10000 --energies 10000 --particles e- --sim
    # gamma, 10 GeV, 90 degrees, 10 events
    # python runParallel.py --outDir $runname/simulation --nEvt 10 --energies 10000 --particles gamma --sim
    # pi-, 10 GeV, 90 degrees, 10 events
    # python runParallel.py --outDir $runname/simulation --nEvt 10 --energies 10000 --particles pi- --sim
    # gamma, 10 GeV, 155-170 degrees, 10 events
    python runParallel.py --outDir $runname/simulation --nEvt 10 --energies 10000 --particles gamma --thetaMinMax 155 170 --sim
fi

if (( $doReconstruction > 0 )); then
    # gamma and pi0, 20 GeV, 90 degrees, 1k events
    # python runParallel.py --outDir $runname/reconstruction --nEvt 1000 --energies 20000 --particles gamma pi0 --reco --simInput $runname/simulation
    # e-, 10 GeV, 90 degrees, 10k events
    # python runParallel.py --outDir $runname/reconstruction --nEvt 10000 --energies 10000 --particles e- --reco --simInput $runname/simulation
    # gamma, 10 GeV, 90 degrees, 10 events
    # python runParallel.py --outDir $runname/reconstruction --nEvt 10 --energies 10000 --particles gamma --reco --simInput $runname/simulation
    # pi-, 10 GeV, 90 degrees, 10 events
    # python runParallel.py --outDir $runname/reconstruction --nEvt 10 --energies 10000 --particles pi- --reco --simInput $runname/simulation --includeHCal
    # gamma, 10 GeV, 155-170 degrees, 10 events
    python runParallel.py --outDir $runname/reconstruction --nEvt 10 --energies 10000 --particles gamma --reco --simInput $runname/simulation --thetaMinMax 155 170
fi


# Generic simulation and reconstruction big production jobs
if (( $doProductionSim > 0 )); then
    python runParallel.py --outDir $runname/production_simulation --nEvt 100000 --particles gamma pi0 --simprod
fi

if (( $doProductionReco > 0 )); then
    python runParallel.py --outDir $runname/production_reconstruction --nEvt 100000 --particles gamma pi0 --recprod --simInput $runname/production_simulation
fi

