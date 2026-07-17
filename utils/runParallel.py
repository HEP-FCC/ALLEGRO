#!/usr/bin/env python

# runParallel.py
# run many jobs in parallel on the multiple cores of the local machine

import os
import os.path
import subprocess
import multiprocessing as mp
import argparse
import json
from math import radians, log10, ceil
import glob
import ROOT

# try to solve disk issues by randomizing order of job submission to avoid many short jobs finishing at the same time
import random

# base name of xml file for ECal barrel - should be the same in ALLEGRO_o1_v03.xml and in run_all_chain.sh
xmlfileECal="ECalBarrel_thetamodulemerged"

debug = False

# set this to True if you want the values of the sampling and upstream corrections
# to be updated in the scripts once they have been calculated. If False, they will
# just be saved in json files but the python steering scripts wont be updated
updateCalibrationsInPython = False

# set this to True if you want the values of the sampling and upstream corrections
# to be read from the JSON files in the working directory.
# If False, the values in the python scripts will be used instead
useUpdatedCalibrations = False

# set this to 0 to be able to reproduce the results, otherwise set it to something
# initialised from current date/time to produce multiple independent samples to analyse
# together when finished (i.e. for very big productions split in smaller productions)
initialSeed = 1
if initialSeed != 0:
    import random
    initialSeed = random.randint(0, 2**24)

saveRawHitInfo = False    # save or not G4 hit info (E/time/position)

nCores = 64

extraRecoArgs = ""

# thetaMin, thetaMax in case a flat theta spectrum is desired (set downstream)
thetaMin = None
thetaMax = None

def executeCmd(cmd):
    print("Running", cmd)
    if not debug:
        return subprocess.run([cmd], shell=True)

def appendToFile(filename, text):
    with open(filename, "a") as file:
        file.write(text + "\n")

class JobProcessor:
    def __init__(self, script, outdir, output_tag, nevts):
        self.script = script
        self.outdir = outdir
        self.output_tag = output_tag
        self.extra_args = ""
        self.isSim = False
        self.dopreprocess = False
        self.dopostprocess = False
        self.nevts = nevts
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(outdir+"/log", exist_ok=True)
        os.makedirs(outdir+"/root", exist_ok=True)

    # the script to process differs for sim and reco, so will be defined in derived classes
    def process(self, nevt, skipevt, energy, theta, particle, jobId):
        return 0

    # default hadd action - hadd all subjobs for given particle, energy, theta
    def hadd(self, energy, theta, particle):
        if theta>0:
            thetaMinRad = radians(theta)
            thetaMaxRad = thetaMinRad
            thetaStr = f"theta_{theta}"
        else:
            thetaMinRad = radians(thetaMin)
            thetaMaxRad = radians(thetaMax)
            thetaStr = f"thetaMinMax_{thetaMin:.1f}_{thetaMax:.1f}"
        filestub = f"{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}"
        # cmd = f"hadd -f {self.outdir}/{filestub}.root {self.outdir}/root/{filestub}_jobid_*.root"
        cmd = f"LD_PRELOAD=maxTreeSize_C.so hadd -f {self.outdir}/{filestub}.root {self.outdir}/root/{filestub}_jobid_*.root"
        # only execute if output file does not exist yet
        if os.path.exists(f"{self.outdir}/{filestub}.root"):
            print(f"Output file {self.outdir}/{filestub}.root already existing, skipping hadd")
        return executeCmd(cmd)

    # default rm action: remove all subjobs for given particle, energy, theta if merged file is OK
    def rm(self, energy, theta, particle):
        if theta>0:
            thetaMinRad = radians(theta)
            thetaMaxRad = thetaMinRad
            thetaStr = f"theta_{theta}"
        else:
            thetaMinRad = radians(thetaMin)
            thetaMaxRad = radians(thetaMax)
            thetaStr = f"thetaMinMax_{thetaMin:.1f}_{thetaMax:.1f}"

        print("Checking output file")
        filestub = f"{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}"
        result = subprocess.run(["../ALLEGRO/util/checkFile.sh", f"{self.outdir}/{filestub}.root", f"{self.nevts}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout.splitlines()[0]
        print(output)
        if output=="OK":
            print(f"Merged file is OK, deleting intermediate files in {self.outdir}/root")
            cmd = f"rm -f {self.outdir}/root/{filestub}_jobid_*.root"
            return executeCmd(cmd)
        else:
            print(f"Merged file not OK, intermediate files in {self.outdir}/root not deleted!")
            # remove merged file
            cmd = f"rm -f {self.outdir}/{filestub}.root"
            return executeCmd(cmd)
            return 0


class SimulationJobProcessor(JobProcessor):
    def __init__(self, outdir, nevts):
        script = "UNUSED"
        output_tag = "simulation"
        super().__init__(script, outdir, output_tag, nevts)
        self.isSim = True
        self.dopreprocess = False
        self.dopostprocess = False

        # add extra argument to turn on detailed shower mode if we want to save G4 hit positions
        if saveRawHitInfo:
            self.extra_args = "--enableDetailedShowerMode"

    # run ddsim
    def process(self, nevt, skipevt, energy, theta, particle, jobId):
        if theta>0:
            thetaMinRad = radians(theta)
            thetaMaxRad = thetaMinRad
            thetaStr = f"theta_{theta}"
        else:
            thetaMinRad = radians(thetaMin)
            thetaMaxRad = radians(thetaMax)
            thetaStr = f"thetaMinMax_{thetaMin:.1f}_{thetaMax:.1f}"
        seed = initialSeed + jobId
        # only execute command if previous attempt was unsuccessul
        if os.path.exists("{self.outdir}/root/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}.root"):
            print(f"merged sim file already produced: {self.outdir}/root/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}.root")
            return True
        if os.path.exists(f"{self.outdir}/root/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}.root"):
            f = ROOT.TFile.Open(f"{self.outdir}/root/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}.root")
            if (f and (t := f.Get("events")) and t.GetEntries()==nevt):
                print(f"unmerged sim file already produced: {self.outdir}/root/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}.root")
                return True
        #
        cmd = f"ddsim --enableGun --gun.distribution uniform --gun.momentumMin {energy}*MeV --gun.momentumMax {energy}*MeV --gun.thetaMin {thetaMinRad} --gun.thetaMax {thetaMaxRad} --gun.particle {particle} --numberOfEvents {nevt} --outputFile {self.outdir}/root/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}.root --random.enableEventSeed --random.seed {seed} --compactFile $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml {self.extra_args} &> {self.outdir}/log/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}.log"
        return executeCmd(cmd)

    # not sure we need to explicitly implement this?
    def hadd(self, energy, theta, particle):
        super().hadd(energy, theta, particle)

    # not sure we need to explicitly implement this?
    def rm(self, energy, theta, particle):
        super().rm(energy, theta, particle)

#    def postprocess(self, energy, theta, particle):
#        return 0
#
#    def postprocess_glob(self):
#        return 0


class ReconstructionJobProcessor(JobProcessor):

    def __init__(self, outdir, siminput, nevts, sampling_fracs=None, corrections=None):
        script = "%s/digi_reco.py" % outdir
        output_tag = "reconstruction"
        self.siminput = siminput
        super().__init__(script, outdir, output_tag, nevts)
        self.isSim = False
        self.dopreprocess = True
        self.dopostprocess = False
        self.sampling_fracs = sampling_fracs
        self.corrections = corrections
        self.extra_args = extraRecoArgs
        # to use custom neighbours and noise maps
        print(f"WARNING: will use reconstruction files from folder {outdir}/../data")
        self.extra_args += f" --dataFolder {outdir}/../data/"

    # before launching reconstruction, needs to preprocess steering file
    def preprocess(self):
        cmd = f"cp -f run_ALLEGRO_reco.py {self.script}"
        executeCmd(cmd)

        if self.sampling_fracs:
            sampling_str = ', '.join([str(s) for s in self.sampling_fracs])
            cmd = f"sed -i 's/ecalBarrelSamplingFraction =.*/ecalBarrelSamplingFraction = [{sampling_str}]/' {self.script}"
            executeCmd(cmd)

        if self.corrections:
            upstream_str = ', '.join([str(s) for s in self.corrections['up']])
            downstream_str = ', '.join([str(s) for s in self.corrections['do']])
            cmd = f"sed -i 's/ecalBarrelUpstreamParameters =.*/ecalBarrelUpstreamParameters = [[{upstream_str}]]/' {self.script}"
            executeCmd(cmd)
            cmd = f"sed -i 's/ecalBarrelDownstreamParameters =.*/ecalBarrelDownstreamParameters = [[{downstream_str}]]/' {self.script}"
            executeCmd(cmd)

        cmd = f"sed -i 's/saveHits *=.*/saveHits = False/' {self.script}"
        executeCmd(cmd)
        cmd = f"sed -i 's/saveCells *=.*/saveCells = False/' {self.script}"
        return executeCmd(cmd)

    def process(self, nevt, skipevt, energy, theta, particle, jobId):
        if theta>0:
            thetaMinRad = radians(theta)
            thetaMaxRad = thetaMinRad
            thetaStr = f"theta_{theta}"
        else:
            thetaMinRad = radians(thetaMin)
            thetaMaxRad = radians(thetaMax)
            thetaStr = f"thetaMinMax_{thetaMin:.1f}_{thetaMax:.1f}"
        cmd = f"k4run {self.script} --IOSvc.FirstEventEntry {skipevt} -n {nevt} --IOSvc.Output {self.outdir}/root/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}.root  --IOSvc.Input {self.siminput}/simulation_energy_{energy}_{thetaStr}_particle_{particle}.root {self.extra_args} &> {self.outdir}/log/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}.log"
        return executeCmd(cmd)

    def hadd(self, energy, theta, particle):
        super().hadd(energy, theta, particle)

    def rm(self, energy, theta, particle):
        super().rm(energy, theta, particle)


class SamplingJobProcessor(JobProcessor):
    def __init__(self, outdir, nevts):
        script = "../ALLEGRO/sampling_fractions/fcc_ee_samplingFraction_inclinedEcal.py"
        output_tag = "sampling_output"
        super().__init__(script, outdir, output_tag, nevts)
        self.isSim = True
        self.dopreprocess = False
        self.dopostprocess = True
        # fiducial - full shower should contained
        #thetaMin = radians(45.)
        #thetaMax = radians(135.)
        #thetaStr = f"thetaMinMax_45.0_135.0"
        # photon direction should be fully contained
        #thetaMin = radians(40.)
        #thetaMax = radians(140.)
        #thetaStr = f"thetaMinMax_40.0_140.0"
        # photon direction should be within ECAL barrel acceptance
        #thetaMin = radians(35.)
        #thetaMax = radians(145.)
        #thetaStr = f"thetaMinMax_35.0_145.0"
        # for comparison to Jiashun
        self.thetaMin = 30.
        self.thetaMax = 150.

    def process(self, nevt, skipevt, energy, theta, particle, jobId):
        # note: skipevt is ignored
        thetarad = radians(theta)

        # first, run the simulation
        if theta>0:
            thetaMin = radians(theta)
            thetaMax = thetaMin
            thetaStr = f"theta_{theta:.1f}"
        else:
            thetaMin = radians(self.thetaMin)
            thetaMax = radians(self.thetaMax)
            thetaStr = f"thetaMinMax_{thetaMin:.1f}_{thetaMax:.1f}"
        cmd = f"ddsim --enableGun --gun.distribution uniform --gun.momentumMin {energy}*MeV --gun.momentumMax {energy}*MeV --gun.thetaMin {thetaMin} --gun.thetaMax {thetaMax} --gun.particle {particle} --numberOfEvents {nevt} --outputFile {self.outdir}/root/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}_sim.root --random.enableEventSeed --random.seed {jobId} --compactFile $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/DectEmptyMaster.xml $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/{xmlfileECal}_calibration.xml {self.extra_args} &> {self.outdir}/log/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}_sim.log"
        result = executeCmd(cmd)
        if result.returncode != 0:
            return result

        # then, run the reconstruction
        jobinput = f"{self.outdir}/root/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}_sim.root"
        joboutput = f"{self.outdir}/root/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}_rec.root"
        cmd = f"""k4run {self.script} -n {nevt} --hists.energyAxis {energy/1000} --IOSvc.Input {jobinput} --Output.THistSvc "rec DATAFILE='{self.outdir}/root/calibration_{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}.root' TYP='ROOT' OPT='RECREATE'" --IOSvc.Output {joboutput} {self.extra_args} &> {self.outdir}/log/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_{jobId}_rec.log"""
        return executeCmd(cmd)

    def hadd(self, energy, theta, particle):
        if theta>0:
            thetaMin = radians(theta)
            thetaMax = thetaMin
            thetaStr = f"theta_{theta:.1f}"
        else:
            thetaMin = radians(self.thetaMin)
            thetaMax = radians(self.thetaMax)
            thetaStr = f"thetaMinMax_{thetaMin:.1f}_{thetaMax:.1f}"

        # add the sim files (actually not needed except for debug)
        # filestub = f"{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}"
        # cmd = f"hadd -f {self.outdir}/{filestub}_sim.root {self.outdir}/root/{filestub}_jobid_*_sim.root"
        # result = executeCmd(cmd)
        # if result.returncode != 0:
        #     return result

        # add the rec files (actually not needed except for debug)
        # filestub = f"{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}"
        # cmd = f"hadd -f {self.outdir}/{filestub}_rec.root {self.outdir}/root/{filestub}_jobid_*_rec.root"
        # result = executeCmd(cmd)
        # if result.returncode != 0:
        #     return result

        # add the calib files
        filestub = f"calibration_{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}"
        cmd = f"hadd -f {self.outdir}/{filestub}.root {self.outdir}/root/{filestub}_jobid_*.root"
        return executeCmd(cmd)

    def rm(self, energy, theta, particle):
        # delete the intermediate files - but only if final calib file exists
        if theta>0:
            thetaMin = radians(theta)
            thetaMax = thetaMin
            thetaStr = f"theta_{theta:.1f}"
        else:
            thetaMin = radians(self.thetaMin)
            thetaMax = radians(self.thetaMax)
            thetaStr = f"thetaMinMax_{thetaMin:.1f}_{thetaMax:.1f}"
        calibfile = f"{self.outdir}/calibration_{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}.root"
        if os.path.isfile(calibfile):
            # delete the rec and sim intermediate files
            subprocess.run([f"rm -f {self.outdir}/root/{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_*.root"], shell=True)
            # delete the calib intermediate files
            cmd = f"rm -f {self.outdir}/root/calibration_{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}_jobid_*.root"
            return executeCmd(cmd)
        else:
            print("Calibration file %s does not exist, will not remove intermediate files to help debugging")
            return -1


    def postprocess(self, energy, theta, particle):
        # make plots of sampling fraction, save SFs to json, and update the values in the python scripts
        if theta>0:
            thetaMin = radians(theta)
            thetaMax = thetaMin
            thetaStr = f"theta_{theta:.1f}"
        else:
            thetaMin = radians(self.thetaMin)
            thetaMax = radians(self.thetaMax)
            thetaStr = f"thetaMinMax_{thetaMin:.1f}_{thetaMax:.1f}"
        energy_GeV = int(energy / 1.e3)
        cmd = f"python ../ALLEGRO/sampling_fractions/FCC_calo_analysis_cpp/plot_samplingFraction.py \
            {self.outdir}/calibration_{self.output_tag}_energy_{energy}_{thetaStr}_particle_{particle}.root {energy_GeV} \
            --preview -outputfolder {self.outdir} --noFits --json {self.outdir}/SF.json"
        if updateCalibrationsInPython:
            cmd += " --sed"
        return executeCmd(cmd)

    def postprocess_glob(self):
        return 0


class UpstreamJobProcessor(JobProcessor):
    def __init__(self, outdir, nevts, sampling_fracs=None, postprocess_scripts_dir=None):
        script = "../ALLEGRO/upstream/fcc_ee_upstream_inclinedEcal.py"
        output_tag = "upstream_output"
        super().__init__(script, outdir, output_tag, nevts)
        self.isSim = True
        self.dopreprocess = False
        self.dopostprocess = True

        if sampling_fracs:
            self.extra_args += "--samplingFractions "
            self.extra_args += ' '.join([str(s) for s in sampling_fracs])
            self.extra_args += ' '
        self.postprocess_dir = os.path.join(os.getenv("FCCBASEDIR"), 'k4SimGeant4/Detector/DetStudies/scripts/')

    def process(self, nevt, skipevt, energy, theta, particle, jobId):
        # note: skipevt is ignored
        thetarad = radians(theta)

        # first, run the simulation
        if theta>0:
            thetaMin = radians(theta)
            thetaMax = thetaMin
        else:
            thetaMin = radians(85.)
            thetaMax = radians(95.)
        cmd = f"ddsim --enableGun --gun.distribution uniform --gun.momentumMin {energy}*MeV --gun.momentumMax {energy}*MeV --gun.thetaMin {thetaMin} --gun.thetaMax {thetaMax} --gun.particle {particle} --numberOfEvents {nevt} --outputFile {self.outdir}/root/{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}_jobid_{jobId}_sim.root --random.enableEventSeed --random.seed {jobId} --compactFile $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/DectEmptyMaster.xml $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/{xmlfileECal}_upstream.xml &> {self.outdir}/log/{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}_jobid_{jobId}_sim.log"
        result = executeCmd(cmd)
        if result.returncode != 0:
            print(result)
        # commented: I have seen this return even when jobs were fine and reco was not run..
        #    return result

        # then, run the reconstruction
        cmd = f"k4run {self.script} -n {nevt} --IOSvc.Input {self.outdir}/root/{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}_jobid_{jobId}_sim.root --IOSvc.Output {self.outdir}/root/{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}_jobid_{jobId}_rec.root {self.extra_args} &> {self.outdir}/log/{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}_jobid_{jobId}_rec.log"
        return executeCmd(cmd)

    def hadd(self, energy, theta, particle):
        filestub = f"{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}"

        # add the sim files (actually not needed except for debug)
        # cmd = f"hadd -f {self.outdir}/{filestub}_sim.root {self.outdir}/root/{filestub}_jobid_*_sim.root"
        # executeCmd(cmd)

        # add the rec files
        cmd = f"hadd -f {self.outdir}/{filestub}_rec.root {self.outdir}/root/{filestub}_jobid_*_rec.root"
        return executeCmd(cmd)

    def rm(self, energy, theta, particle):
        outfile = f"{self.outdir}/{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}_rec.root"
        if os.path.isfile(outfile):
            # delete the rec and sim intermediate files
            cmd = f"rm -f {self.outdir}/root/{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}_jobid_*.root"
            return executeCmd(cmd)
        else:
            print(f"Merged output file {outfile} does not exist, will not remove intermediate files to help debugging")
            return -1

    def postprocess(self, energy, theta, particle):
        # for each energy point, determine (fit) the dependence of energy upstream vs energy in 1st layer
        emax = 0.015 * (1. + log10(energy / 1000.))
        emin = 0.002 * (1. + log10(energy / 1000.))
        # cmd = f"{self.postprocess_dir}/cec_process_events -i {self.outdir}/{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}_rec.root -t upstream --plot-file-format png -o {self.outdir}/fit_results.json --plot-directory {self.outdir} --func-from {emin} --func-to {emax}"
        cmd = f"../ALLEGRO/upstream/cec_process_events -i {self.outdir}/{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}_rec.root -t upstream --plot-file-format png -o {self.outdir}/fit_results.json --plot-directory {self.outdir} --func-from {emin} --func-to {emax}"
        executeCmd(cmd)

        # for each energy point, determine (fit) the dependence of energy downstream vs energy in last layer
        emax = 0.04 * energy / 1000
        if energy >= 10000:
            emax *= (log10(energy / 1000.))
        emin = 0.005
        # cmd = f"{self.postprocess_dir}/cec_process_events -i {self.outdir}/{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}_rec.root -t downstream --function pol2 --plot-file-format png -o {self.outdir}/fit_results.json --plot-directory {self.outdir} --func-from {emin} --func-to {emax}"
        cmd = f"../ALLEGRO/upstream/cec_process_events -i {self.outdir}/{self.output_tag}_energy_{energy}_theta_{theta}_particle_{particle}_rec.root -t downstream --function pol2 --plot-file-format png -o {self.outdir}/fit_results.json --plot-directory {self.outdir} --func-from {emin} --func-to {emax}"
        return executeCmd(cmd)

    def postprocess_glob(self):
        # fit the dependence of the coefficients of the upstream corrections on energy and save to json
        # cmd = f"{self.postprocess_dir}/cec_derive1 -i {self.outdir}/fit_results.json -t upstream --plot-file-format png --plot-directory {self.outdir} --functions '[0]+[1]/(x-[2])' '[0]+[1]/(x-[2])'"
        cmd = f"../ALLEGRO/upstream/cec_derive1 -i {self.outdir}/fit_results.json -t upstream --plot-file-format png --plot-directory {self.outdir} --functions '[0]+[1]/(x-[2])' '[0]+[1]/(x-[2])'"
        executeCmd(cmd)

        # fit the dependence of the coefficients of the downsstream corrections on energy and save to json
        # cmd = f"{self.postprocess_dir}/cec_derive1 -i {self.outdir}/fit_results.json -t downstream --plot-file-format png --plot-directory {self.outdir} --functions '[0]+[1]*x' '[0]+[1]/sqrt(x)' '[0]+[1]/x'"
        cmd = f"../ALLEGRO/upstream/cec_derive1 -i {self.outdir}/fit_results.json -t downstream --plot-file-format png --plot-directory {self.outdir} --functions '[0]+[1]*x' '[0]+[1]/sqrt(x)' '[0]+[1]/x'"
        executeCmd(cmd)

        # update the values of the parameters in the python scripts
        if updateCalibrationsInPython:
            cmd = f"python ../ALLEGRO/upstream/read_upstream_json.py {self.outdir}/corr_params_1d.json"
            return executeCmd(cmd)
        else:
            return 0


class ProductionJobProcessor():
    def __init__(self, script, outdir, output_tag, nevts):
        # script = "NONE"  # unused
        # super().__init__(script, outdir, output_tag)
        self.script = script
        self.outdir = outdir
        self.output_tag = output_tag
        self.extra_args = ""
        self.isSim = False
        self.dopreprocess = False
        self.dopostprocess = False
        self.nevts = nevts
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(outdir+"/log", exist_ok=True)
        os.makedirs(outdir+"/root", exist_ok=True)

    def process(self, nevt, skipevt, particle, jobId):
        # needs to be implemented in derived classes
        return 0

    def hadd(self, particle):
        filestub = f"{self.output_tag}_particle_{particle}"
        # cmd = f"hadd -f {self.outdir}/{filestub}.root {self.outdir}/root/{filestub}_jobid_*.root"
        # workaround to handle > 100 GB trees (hadd will crash otherwise, see
        # https://root-forum.cern.ch/t/root-6-04-14-hadd-100gb-and-rootlogon/24581
        # in alternative do not merge and use a TChain instead
        cmd = f"LD_PRELOAD=maxTreeSize_C.so hadd -f {self.outdir}/{filestub}.root {self.outdir}/root/{filestub}_jobid_*.root"
        return executeCmd(cmd)

    def rm(self, particle):
        print("Checking output file")
        result = subprocess.run(["../ALLEGRO/util/checkFile.sh", f"{self.outdir}/{self.output_tag}_particle_{particle}.root", f"{self.nevts}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout.splitlines()[0]
        print(output)
        # in that case delete the subjob output
        if output=="OK":
            print(f"Merged file is OK, deleting intermediate files in {self.outdir}/root")
            cmd = f"rm -f {self.outdir}/root/{self.output_tag}_particle_{particle}_jobid_*.root"
            return executeCmd(cmd)
        else:
            print(f"Merged file not OK, intermediate files in {self.outdir}/root not deleted!")
            return 0

        cmd = f"rm -f {self.outdir}/root/{self.output_tag}_particle_{particle}_jobid_*.root"
        # TODO: do it only if merged root file exists and can be opened
    #    return executeCmd(cmd)


class SimulationProductionJobProcessor(ProductionJobProcessor):

    def __init__(self, outdir, nevts):
        output_tag = "production_simulation"
        script = "ddim"
        super().__init__(script, outdir, output_tag, nevts)
        self.isSim = True
        self.dopreprocess = False
        self.dopostprocess = False
        # add extra argument to turn on detailed shower mode if we want to save G4 hit positions
        if saveRawHitInfo:
            self.extra_args = "--enableDetailedShowerMode"

    # no preprocessing needed

    # process: run ddsim
    def process(self, nevt, skipevt, particle, jobId):
        # for barrel
        # thetamin = radians(40.)
        # thetamax = radians(140.)
        # for endcap
        # thetamin = radians(5.)
        # thetamax = radians(40.)
        # do not use hardcoded value, rather take them from command line
        thetamin = radians(thetaMin)
        thetamax = radians(thetaMax)
        energymin = 100
        energymax = 105000
        # energymax = 22000
        # energymin = 105000
        # energymax = 200000
        seed = initialSeed + jobId

        # cmd = f"ddsim --enableGun --gun.distribution uniform --gun.momentumMin {energymin}*MeV --gun.momentumMax {energymax}*MeV --gun.thetaMin {thetamin} --gun.thetaMax {thetamax} --gun.particle {particle} --numberOfEvents {nevt} --outputFile {self.outdir}/root/{self.output_tag}_particle_{particle}_jobid_{jobId}.root --random.enableEventSeed --random.seed {seed} --compactFile $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml {self.extra_args} &> {self.outdir}/log/{self.output_tag}_particle_{particle}_jobid_{jobId}.log"

        # first, create the steering file
        executeCmd(f"mkdir -p {self.outdir}/python")
        steeringScript = f"{self.outdir}/python/{self.output_tag}_particle_{particle}_jobid_{jobId}_steersim.py"
        cmd = f"ddsim --enableGun --gun.distribution uniform --gun.momentumMin {energymin}*MeV --gun.momentumMax {energymax}*MeV --gun.thetaMin {thetamin} --gun.thetaMax {thetamax} --gun.particle {particle} --numberOfEvents {nevt} --outputFile {self.outdir}/root/{self.output_tag}_particle_{particle}_jobid_{jobId}.root --random.enableEventSeed --random.seed {seed} --compactFile $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml {self.extra_args} --dumpSteeringFile > {steeringScript}"
        executeCmd(cmd)

        # hack steering file if we need to drop some collection
        dropDetectorHits = ["VertexBarrel", "VertexEndcap", "DCH_v2", "SiWrB", "SiWrD", "MuonTaggerBarrel", "MuonTaggerEndcap_positive", "MuonTaggerEndcap_negative", "LumiCal"]
        if len(dropDetectorHits)>0:
            print("WARNING: will not save detector hits for the following collections")
            print(dropDetectorHits)
            appendToFile(steeringScript, '')
            appendToFile(steeringScript, '## Custom filters')
            appendToFile(steeringScript, 'SIM.filter.filters["rejectCollection"] = dict(name="EnergyDepositMinimumCut/1TeV", parameter={"Cut": 1000.0*GeV})')
        for det in dropDetectorHits:
            appendToFile(steeringScript, 'SIM.filter.mapDetFilter["%s"] = "rejectCollection"' % det)

        # run sim
        cmd = f"ddsim --steeringFile {steeringScript} &> {self.outdir}/log/{self.output_tag}_particle_{particle}_jobid_{jobId}.log"
        return executeCmd(cmd)


class ReconstructionProductionJobProcessor(ProductionJobProcessor):

    def __init__(self, outdir, siminput, nevts, sampling_fracs=None, corrections=None):
        output_tag = "production_reconstruction"
        script = "%s/digi_reco.py" % outdir
        super().__init__(script, outdir, output_tag, nevts)
        self.outdir = outdir
        self.siminput = siminput
        self.isSim = False
        self.dopreprocess = True
        self.dopostprocess = False
        self.sampling_fracs = sampling_fracs
        self.corrections = corrections
        self.extra_args = extraRecoArgs
        # to use custom neighbours and noise maps
        print(f"WARNING: will use reconstruction files from folder {outdir}/../data")
        self.extra_args += f" --dataFolder {outdir}/../data/"

    # before launching reconstruction, needs to preprocess steering file
    def preprocess(self):
        cmd = "cp -f run_ALLEGRO_reco.py %s/digi_reco.py" % self.outdir
        executeCmd(cmd)

        if self.sampling_fracs:
            sampling_str = ', '.join([str(s) for s in self.sampling_fracs])
            cmd = f"sed -i 's/ecalBarrelSamplingFraction =.*/ecalBarrelSamplingFraction = [[{sampling_str}]]/' {self.script}"
            executeCmd(cmd)

        if self.corrections:
            upstream_str = ', '.join([str(s) for s in self.corrections['up']])
            downstream_str = ', '.join([str(s) for s in self.corrections['do']])
            cmd = f"sed -i 's/ecalBarrelUpstreamParameters =.*/ecalBarrelUpstreamParameters = [[{upstream_str}]]/' {self.script}"
            executeCmd(cmd)
            cmd = f"sed -i 's/ecalBarrelDownstreamParameters =.*/ecalBarrelDownstreamParameters = [[{downstream_str}]]/' {self.script}"
            executeCmd(cmd)

        cmd = f"sed -i 's/saveHits *=.*/saveHits = False/' {self.script}"
        executeCmd(cmd)
        cmd = f"sed -i 's/saveCells *=.*/saveCells = False/' {self.script}"
        executeCmd(cmd)

    # process: run k4run
    def process(self, nevt, skipevt, particle, jobId):
        cmd = f"k4run {self.outdir}/digi_reco.py --IOSvc.FirstEventEntry {skipevt} -n {nevt} --IOSvc.Output {self.outdir}/root/{self.output_tag}_particle_{particle}_jobid_{jobId}.root  --IOSvc.Input {self.siminput}/production_simulation_particle_{particle}.root {self.extra_args} &> {self.outdir}/log/{self.output_tag}_particle_{particle}_jobid_{jobId}.log"
        return executeCmd(cmd)

    def processNInputFiles(self, nInputFiles, skipInputFiles, particle, jobId):
        cmd = f"k4run {self.outdir}/digi_reco.py --IOSvc.Output {self.outdir}/root/{self.output_tag}_particle_{particle}_jobid_{jobId}.root  --IOSvc.Input"
        file_pattern = os.path.join(f"{self.siminput}/root/", f"production_simulation_particle_{particle}_jobid_*.root")
        #print(file_pattern)
        matching_files = glob.glob(file_pattern)
        #print(matching_files)
        for inputFile in range(skipInputFiles, nInputFiles+skipInputFiles):
            cmd += (" " + matching_files[inputFile])
        cmd += f" {self.extra_args} &> {self.outdir}/log/{self.output_tag}_particle_{particle}_jobid_{jobId}.log"
        return executeCmd(cmd)

    def rm(self, particle):
        # check that reconstruction file is OK
        # would be good to check nevents too
        print("Checking output file")
        result = subprocess.run(["../ALLEGRO/util/checkFile.sh", f"{self.outdir}/{self.output_tag}_particle_{particle}.root", f"{self.nevts}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout.splitlines()[0]
        print(output)
        # in that case delete the subjob output
        if output=="OK":
            print(f"Merged file is OK, deleting intermediate files in {self.outdir}/root")
            cmd = f"rm -f {self.outdir}/root/{self.output_tag}_particle_{particle}_jobid_*.root"
            return executeCmd(cmd)
        else:
            print(f"Merged file not OK, intermediate files in {self.outdir}/root not deleted!")
            return 0


def run_the_jobs(jobProcessor, particles, energies, thetas, nEvt, do_preprocess, do_process, do_postprocess, issim):
    if do_preprocess:
        print("Doing preprocessing")
        jobProcessor.preprocess()

    print("nCores = ", nCores)
    with mp.Pool(processes=nCores) as p:
        print("Pool size =", p._processes)
        # simu is much slower than reconstruction so there are more jobs needed; also, simu at high E with many more hits is much slower
        if issim:
            nEvtMaxPerJob = [int(210 * (10000. / e)**1.2) for e in energies]  # 210 evts for 10 GeV, with power scaling (so that on 96 cores I can do 20k evts in one go)
        else:
            # nEvtMaxPerJob = [int(3600-1500*log10(e/1000.)) for e in energies]  # 2100 evts for 10 GeV, with log E scaling (on 96 cores => > 200k events at 10 GeV). 600 evts/job at 100 GeV, 3600 at 1 GeV
            nEvtMaxPerJob = [500 for e in energies]  # reco jobs are similar in speed no matter the energy
        nEvtPerJob = [min(nEvt, nMax) for nMax in nEvtMaxPerJob]
        args = []
        energies_and_thetas_and_particles = []
        jobId = 1
        for particle in particles:
            for theta in thetas:
                for e, nEvtPerJobForE in zip(energies, nEvtPerJob):
                    nEvtToLaunch = nEvt
                    while nEvtToLaunch > 0:
                        nLaunched = min(nEvtToLaunch, nEvtPerJobForE)
                        args.append((nLaunched, nEvt-nEvtToLaunch, e, theta, particle, jobId))
                        jobId += 1
                        nEvtToLaunch -= nLaunched
                for e in energies:
                    energies_and_thetas_and_particles.append((e, theta, particle))

        if do_process:
            print("About to send jobs with parameters:")
            for a in args:
                print(a)

            random.shuffle(args)
            res = p.starmap_async(jobProcessor.process, args)
            print("Retcodes of the jobs:")
            print(res.get())
            print()

            print("Hadd'ing results:")
            res = p.starmap_async(jobProcessor.hadd, energies_and_thetas_and_particles)
            print("Retcodes of the jobs:")
            print(res.get())
            print()

            print("Removing intermediate files:")
            res = p.starmap_async(jobProcessor.rm, energies_and_thetas_and_particles)
            print("Retcodes of the jobs:")
            print(res.get())
            print()

        if do_postprocess:
            print("Doing postprocessing")
            # Do it sequentially
            res = p.starmap(jobProcessor.postprocess, energies_and_thetas_and_particles, chunksize=50)
            jobProcessor.postprocess_glob()


def run_production(jobProcessor, particles, nEvt, nEvtMaxPerJob, do_preprocess, do_process, issim):
    if do_preprocess:
        print("Doing preprocessing")
        jobProcessor.preprocess()

    print("nCores = ", nCores)
    with mp.Pool(processes=nCores) as p:
        print("Pool size =", p._processes)
        args = []
        jobId = 1
        if nEvt>0:
            nEvtPerJob = min(nEvt, nEvtMaxPerJob)
            for particle in particles:
                nEvtToLaunch = nEvt
                while nEvtToLaunch > 0:
                    nLaunched = min(nEvtToLaunch, nEvtPerJob)
                    args.append((nLaunched, nEvt-nEvtToLaunch, particle, jobId))
                    jobId += 1
                    nEvtToLaunch -= nLaunched
        else:
            for particle in particles:
                # nEvt is number of input files and nEvtMaxPerJob is number of files per job
                nFiles = -nEvt
                nFilesPerJob = -nEvtMaxPerJob
                nFilesToLaunch = nFiles
                while nFilesToLaunch > 0:
                    nLaunched = min(nFilesToLaunch, nFilesPerJob)
                    args.append((nLaunched, nFiles-nFilesToLaunch, particle, jobId))
                    jobId += 1
                    nFilesToLaunch -= nLaunched

        if do_process:
            print("About to send jobs with parameters:")
            for a in args:
                print(a)
            if nEvt > 0:
                res = p.starmap_async(jobProcessor.process, args)
            else:
                res = p.starmap_async(jobProcessor.processNInputFiles, args)
            print("Retcodes of the jobs:")
            print(res.get())
            print()

            print("Hadd'ing results:")
            res = p.map_async(jobProcessor.hadd, particles)
            print("Retcodes of the jobs:")
            print(res.get())
            print()

            print("Removing intermediate files:")
            res = p.map_async(jobProcessor.rm, particles)
            print("Retcodes of the jobs:")
            print(res.get())
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outDir', default='./', type=str, help='output directory for plots')
    parser.add_argument('--nEvt', default=1000, type=int, help='number of events to process per point')
    parser.add_argument('--energies', default=[], action='extend', nargs='*', type=int, help='energies to process')
    theta_group = parser.add_mutually_exclusive_group()
    theta_group.add_argument('--thetas', default=None, action='extend', nargs='*', type=int, help='thetas (in degrees) to process. If not supplied, will use 90 degrees')
    theta_group.add_argument('--thetaMinMax', default=None, nargs=2, type=float, metavar=('THETA_MIN', 'THETA_MAX'), help='thetas (in degrees) to process. If not supplied, will use 90 degrees')
    parser.add_argument('--particles', default=[], action='extend', nargs='*', type=str, help='particles to process. If not supplied, will use electrons')
    parser.add_argument('--process', default=True, action=argparse.BooleanOptionalAction, help='if no-process is set, do only the pre- and post-processing - no simulation jobs are run')
    parser.add_argument('--inputFilesPerJob', default=-1, type=int, help='number of input files per job (only valid for reconstruction jobs - will run over unmerged simulation files)')
    parser.add_argument('--addNoise', action='store_true', help='add noise in reconstruction', default=False)
    parser.add_argument('--addCrosstalk', action='store_true', help='add crosstalk in reconstruction', default=False)
    parser.add_argument('--includeHCal', action='store_true', help='include HCal in reconstruction', default=False)
    parser.add_argument('--addTracks', action='store_true', help='add tracks in reconstruction', default=False)
    parser.add_argument('--simInput', default='', type=str, help='location of the simulation inputs for the reconstruction step')
    parser.add_argument('--SF', default='', type=str, help='JSON file containing sampling fractions')
    parser.add_argument('--corrections', default='', type=str, help='JSON file containing upstream and downstream corrections')
    # parser.add_argument('--decorateClusters', action='store_true', help='add shape parameters to clusters', default=False)  # commented. Lets always decorate clusters
    parser.add_argument('--calibrateClusters', action='store_true', help='apply MVA calibration to clusters', default=False)
    parser.add_argument('--runPhotonID', action='store_true', help='run photon ID algorithm', default=False)
    parser.add_argument('--reconstructPi0s', action='store_true', help='reconstruct resolved pi0 candidates', default=False)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sim', action='store_true', help='run the generation/simulation with ddsim')
    group.add_argument('--reco', action='store_true', help='run the reconstruction on the output of ddsim')
    group.add_argument('--simprod', action='store_true', help='production run of particles of all energies and theta (simulation step)')
    group.add_argument('--recprod', action='store_true', help='production run of particles of all energies and theta (reconstruction step)')
    group.add_argument('--sampling', action='store_true', help='compute the sampling fractions')
    group.add_argument('--upstream', action='store_true', help='compute the upstream corrections')
    group.add_argument('--clusters', action='store_true', help='run fixed size and topo clusterings')

    args = parser.parse_args()

    global extraRecoArgs
    if args.includeHCal:
        extraRecoArgs += " --includeHCal"
    if args.addNoise:
        extraRecoArgs += " --addNoise"
    if args.addCrosstalk:
        extraRecoArgs += " --addCrosstalk"
    if args.addTracks:
        extraRecoArgs += " --addTracks"
    if args.calibrateClusters:
        extraRecoArgs += " --calibrateClusters"
    if args.runPhotonID:
        extraRecoArgs += " --runPhotonID"
    if args.reconstructPi0s:
        extraRecoArgs += " --reconstructPi0s true"
    else:
        extraRecoArgs += " --reconstructPi0s false"

    thetas = None
    global thetaMin, thetaMax
    if args.thetas is None and args.thetaMinMax is None:
        thetas = [90.0]
        thetaMin = 90
        thetaMax = 90
    elif args.thetas is not None:
        thetas = sorted(set(args.thetas))  # remove possible duplicates and sort so that we are deterministic in jobid
    else:
        thetaMin, thetaMax = args.thetaMinMax
        thetas = [-1]

    # remove possible duplicates and sort so that we are deterministic in jobid
    energies = sorted(set(args.energies))
    particles = sorted(set(args.particles))
    # default value for energy if not provided
    if len(particles)==0:
        particles=["e-"]

    sampling_fracs = None
    if args.SF!="":
        if not useUpdatedCalibrations:
            print("useUpdatedCalibrations is False, will ignore SFs from file:", args.SF)
        else:
            print("Reading SFs from file:", args.SF)
            with open(args.SF, 'r') as jsonfile:
                data = json.load(jsonfile)
                sampling_fracs = data["SF"]
                print("Applying sampling fractions:", sampling_fracs)

    corrections = None
    if args.corrections!="":
        if not useUpdatedCalibrations:
            print("useUpdatedCalibrations is False, will ignore corrections from file:", args.corrections)
        else:
            print("Reading corrections from file:", args.corrections)
            with open(args.corrections, 'r') as jsonfile:
                source = json.load(jsonfile)
                dict_up = {e['name']: e['value'] for e in source['corr_params'] if e['type'] == 'upstream'}
                dict_do = {e['name']: e['value'] for e in source['corr_params'] if e['type'] == 'downstream'}
                corrections = {}
                corrections['up'] = [dict_up['a'], dict_up['b'], dict_up['c'], dict_up['d'], dict_up['e'], dict_up['f']]
                corrections['do'] = [dict_do['a'], dict_do['b'], dict_do['c'], dict_do['d'], dict_do['e'], dict_do['f']]
                print("Applying up/downstream corrections")
                print(corrections)

    # energies = [500, 1000, 5000, 10000, 15000, 20000, 30000, 50000, 75000, 100000]
    # in principle one would not need to pass to run_the_jobs the pre/post process and isSim info which is already
    # encapsulated in the object passed as 1st parameter. It is left here so that it can be overridden (could add a cmd line option)
    if args.sampling:
        samJobPr = SamplingJobProcessor(args.outDir, args.nEvt)
        run_the_jobs(samJobPr, particles, energies, thetas, args.nEvt, samJobPr.dopreprocess, args.process, samJobPr.dopostprocess, samJobPr.isSim)
    elif args.upstream:
        upJobPr = UpstreamJobProcessor(args.outDir, args.nEvt, sampling_fracs=sampling_fracs)
        run_the_jobs(upJobPr, particles, energies, thetas, args.nEvt, upJobPr.dopreprocess, args.process, upJobPr.dopostprocess, upJobPr.isSim)
    elif args.sim:
        simJobPr = SimulationJobProcessor(args.outDir, args.nEvt)
        run_the_jobs(simJobPr, particles, energies, thetas, args.nEvt, simJobPr.dopreprocess, args.process, simJobPr.dopostprocess, simJobPr.isSim)
    elif args.reco:
        recoJobPr = ReconstructionJobProcessor(args.outDir, args.simInput, args.nEvt, sampling_fracs, corrections)
        run_the_jobs(recoJobPr, particles, energies, thetas, args.nEvt, recoJobPr.dopreprocess, args.process, recoJobPr.dopostprocess, recoJobPr.isSim)
    elif args.clusters:
        # if siminput is not set, then produce the simulation step first:
        if args.simInput=="":
            simJobPr = SimulationJobProcessor(args.outDir, args.nEvt)
            run_the_jobs(simJobPr, particles, energies, thetas, args.nEvt, simJobPr.dopreprocess, args.process, simJobPr.dopostprocess, simJobPr.isSim)
            recoJobPr = ReconstructionJobProcessor(args.outDir, args.outDir, args.nEvt, sampling_fracs, corrections)
        else:
            recoJobPr = ReconstructionJobProcessor(args.outDir, args.simInput, args.nEvt, sampling_fracs, corrections)
        run_the_jobs(recoJobPr, particles, energies, thetas, args.nEvt, recoJobPr.dopreprocess, args.process, recoJobPr.dopostprocess, recoJobPr.isSim)
    elif args.simprod:
        simProdJobPr = SimulationProductionJobProcessor(args.outDir, args.nEvt)
        run_production(simProdJobPr, particles, args.nEvt, min(2000, ceil(args.nEvt*len(particles)/nCores)), simProdJobPr.dopreprocess, args.process, simProdJobPr.isSim)
    elif args.recprod:
        recProdJobPr = ReconstructionProductionJobProcessor(args.outDir, args.simInput, args.nEvt, sampling_fracs=sampling_fracs, corrections=corrections)
        if args.inputFilesPerJob>0:
            # run over the unmerged sim files
            # find out number of total input files
            particle = particles[0]
            file_pattern = os.path.join(f"{args.simInput}/root/", f"production_simulation_particle_{particle}_jobid_*.root")
            matching_files = glob.glob(file_pattern)
            nInputFiles = len(matching_files)
            # launch the production. Pass input input file arguments as negative to be able to distinguish from case in which number of events is passed
            run_production(recProdJobPr, particles, -nInputFiles, -args.inputFilesPerJob, recProdJobPr.dopreprocess, args.process, recProdJobPr.isSim)
        else:
            # run over the merged sim file
            # assume that nEvt passed via CLI contains the proper number of events in the simulation
            run_production(recProdJobPr, particles, args.nEvt, min(4000, ceil(args.nEvt*len(particles)/nCores)), recProdJobPr.dopreprocess, args.process, recProdJobPr.isSim)
    #elif args.production:
    #    clJobPr = ClusterProductionJobProcessor(args.outDir, args.nEvt, sampling_fracs=sampling_fracs, corrections=corrections)
    #    run_production(clJobPr, args.nEvt, True, args.process, False, False)


if __name__ == "__main__":
    main()
