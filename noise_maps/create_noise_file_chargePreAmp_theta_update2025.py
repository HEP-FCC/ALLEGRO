# This script uses the output of create_capacitance_file_theta_update2025.py to create
# a TFile with noise vs theta for the different layers.
#
# Updated in 2025 with new noise estimation from Omega labs
#
# The noise is derived based on https://indico.in2p3.fr/event/37608/contributions/165205/ slide 14

# execute script with
# python create_noise_file_chargePreAmp_theta_update2025.py

# the input file name is configured with capa_filename
# noise and capacitances per layer are saved in root files
# capacitances_ecalBarrelFCCee_theta.root
# elecNoise_ecalBarrelFCCee_theta.root
# in folder noise_capa_<date>

from ROOT import TH1F, TCanvas, TLegend, TFile, gStyle, gPad
import ROOT
import itertools
from datetime import date
import os, sys
#from numpy import ones,vstack
#from numpy.linalg import lstsq
import numpy as np
from math import floor

from scipy.optimize import curve_fit

ROOT.gROOT.SetBatch(ROOT.kTRUE)
ROOT.gStyle.SetPadTickY(1)
ROOT.gStyle.SetOptTitle(0)

def rescaleaxis(g, scale = 50e-12/1e-9):
    """This function rescales the x-axis on a TGraph."""
    N = g.GetN()
    x = g.GetX()
    for i in range(N):
        x[i] *= scale
    g.GetHistogram().Delete()
    g.SetHistogram(0)
    return

# you know the noise charge rms from Martin (as a function of capacitance): https://indico.cern.ch/event/1066234/contributions/4708987/attachments/2387716/4080914/20220209_Brieuc_Francois_Noble_Liquid_Calorimetry_forFCCee_FCCworkshop2022.pdf#page=7
# you need to further get, for each layer, the collected charge corresponding to an energy deposit of 1 MeV in the cell (cell considered including the energy in absorber and PCB), the cell merging strategy does not matter yet here to first approximation because the 1 MeV current equivalent will be the same in any merging scenarios (the current will be shared in more gaps when merging many cells, but all these current will then be 'summed' before to reach the readout). Merging many cells will pay back later, when more signal will be collected per read out channel compared to merging less cells, for the same noise values

# Retrieving a capa dependent function for the noise charge rms in terms of number of electrons: https://indico.cern.ch/event/1066234/contributions/4708987/attachments/2387716/4080914/20220209_Brieuc_Francois_Noble_Liquid_Calorimetry_forFCCee_FCCworkshop2022.pdf#page=7

"""
points_capa_noise = [(100, 4375), (500, 6750)]
x_coords, y_coords = zip(*points_capa_noise)
A = vstack([x_coords, ones(len(x_coords))]).T
m, c = lstsq(A, y_coords,rcond=-1)[0]
#print("Line Solution is y = {m}x + {c}".format(m=m, c=c))
def get_noise_charge_rms(capacitance):
    return m * capacitance + c # number of electrons
"""

# Capa-to-noise relation updated to Omega labs' model in December 2025

# Capacitance (x) and ENC (y) points from Aimie Laffitte's plot
# (three needed but let's take more for a more reliable fit)

x = np.array([100.2, 199.8, 300, 400.2, 500.4, 600])
y = np.array([1020, 1254, 1546, 1878, 2239, 2610])

def parabolic_func(x, a, b):
    return np.sqrt(a * x **2 + b)
    
# Fit the curve
popt, pcov = curve_fit(parabolic_func, x, y)

# Extract optimized parameters
a_opt, b_opt = popt
print(f"Optimal Parameters: a = {a_opt}, b = {b_opt}")

def get_noise_charge_rms(capacitance):
    return np.sqrt( a_opt*capacitance**2 + b_opt ) # number of electrons


# Get the equivalent of 1 MeV energy deposit in a cell (absorber + Lar) in terms of number of electrons in the charge pre-amplifier + shaper
r_recomb = 0.04
w_lar = 23.6 # eV needed to create a ion/electron pair

def get_ref_charge(SF, E_dep = 1 * pow(10, 6)): #E_dep en eV, choose 1 MeV
    return E_dep * SF * (1 - r_recomb) / (2 * w_lar) # nA, the factor 2 comes from: Q_tot = I_0 * t_drift * 1/2  (rectangle --> triangle), t_drift cancels out from the formula to get I_0 which has v_drift/d_gap (Ramo Shockley).
                                                     # Assumption: shaping time is similar or bigger to drift time

# Sampling fraction in each layer
# numbers updated for v03 model with 1536 modules and 11 layers, calculated with ddsim
SFfcc = [0.3800493723322256] * 1 + [0.13494147915064658] * 1 + [0.142866851721152] * 1 + [0.14839315921940666] * 1 + [0.15298362570665006] * 1 + [0.15709704561942747] * 1 + [0.16063717490147533] * 1 + [0.1641723795419055] * 1 + [0.16845490287689746] * 1 + [0.17111520115997653] * 1 + [0.1730605163148862] * 1
nLayers = len(SFfcc)

print("##########: ", get_ref_charge(0.16))

SF_rounded_forPrint = []
for SF in SFfcc:
    SF_rounded_forPrint.append(round(SF,2))
print('SF:', SF_rounded_forPrint)

#output_folder = "noise_capa_" + date.today().strftime("%y%m%d")
output_folder = "noise_vs_capa_chargePreAmp"

#filename = "ecalBarrelFCCee_"+flagImpedance+"Ohm_"+flagTraces+"_"+str(flagsShieldsWidth)+"shieldWidth"
capa_filename = "capacitances_perSource_ecalBarrelFCCee_theta_update2025.root"
if not os.path.exists(capa_filename):
    print("Error: capacitance file does not exist, please run first python create_capacitance_file.py")
    sys.exit(1)
fIn = TFile(capa_filename, "r")

output_folder = "noise_capa_" + date.today().strftime("%y%m%d") 
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
fSaveAll = TFile(os.path.join(output_folder, "capacitances_ecalBarrelFCCee_theta.root"),"RECREATE")
fSave = TFile(os.path.join(output_folder, "elecNoise_ecalBarrelFCCee_theta.root"),"RECREATE")

gStyle.SetOptStat(0)

#TH1 with capacitances
hCapShield = []
hCapTrace = []
hCapDetector = []
hCapTotal = []
# electronic noise histograms
h_elecNoise_fcc = [] # default total noise shield + detector capacitance (without trace capacitance) -> to be used in FCCSW as noise estimation
                     # Update2025: now trace capa included in all histograms!
                     
h_elecNoise_all = [] # total noise shield + trace + detector capacitance
h_elecNoise_withTraceCap = [] # total noise. In 2025 update this is same as above.
h_elecNoise_shield = []
h_elecNoise_trace = []
h_elecNoise_detector = []

#Read graphs from files
for i in range (0, nLayers):  
    nameShield = "hCapacitance_shields"+str(i)
    hCapShield.append(fIn.Get(nameShield))
    nameTrace = "hCapacitance_traces"+str(i)
    hCapTrace.append(fIn.Get(nameTrace))
    nameDetector = "hCapacitance_detector"+str(i)
    hCapDetector.append(fIn.Get(nameDetector))

index = 0    
nbins = hCapShield[index].GetNbinsX()
thetaMin = hCapShield[index].GetXaxis().GetBinLowEdge(1)
thetaMax = hCapShield[index].GetXaxis().GetBinUpEdge(nbins)
print("number of bins ", nbins, ", thetaMin ", thetaMin, ", thetaMax ", thetaMax)

maximumCap = 0.
maximumNoise = 0.
maximumNoiseWithTrace = 0.

line_color_number = 1
line_style_number = 1
for i in range (0, nLayers):  
    if line_color_number == 10:
        line_color_number = 28
    if line_style_number > 10:
        line_style_number = 1
    #Prepare electronic noise histograms    
    h_elecNoise_fcc.append( TH1F() )
    h_elecNoise_fcc[i].SetLineWidth(3)
    h_elecNoise_fcc[i].SetLineColor(line_color_number)
    h_elecNoise_fcc[i].SetLineStyle(line_style_number)
    h_elecNoise_fcc[i].SetBins(nbins, thetaMin, thetaMax)
    h_elecNoise_fcc[i].SetTitle("Default electronic noise: shield + detector + trace capacitance; #theta; Electronic noise [MeV]")
    h_elecNoise_fcc[i].SetName("h_elecNoise_fcc_"+str(i+1))

    h_elecNoise_all.append( TH1F() )
    h_elecNoise_all[i].SetLineWidth(3)
    h_elecNoise_all[i].SetLineColor(line_color_number)
    h_elecNoise_all[i].SetLineStyle(line_style_number)
    h_elecNoise_all[i].SetBins(nbins, thetaMin, thetaMax)
    h_elecNoise_all[i].SetTitle("Total electronic noise: shield + trace + detector capacitance; #theta; Electronic noise [MeV]")
    h_elecNoise_all[i].SetName("h_elecNoise_all_"+str(i+1))
    
    h_elecNoise_withTraceCap.append( TH1F() )
    h_elecNoise_withTraceCap[i].SetLineWidth(3)
    h_elecNoise_withTraceCap[i].SetLineColor(line_color_number)
    h_elecNoise_withTraceCap[i].SetLineStyle(line_style_number)
    h_elecNoise_withTraceCap[i].SetBins(nbins, thetaMin, thetaMax)
    h_elecNoise_withTraceCap[i].SetTitle("Electronic noise with trace capacitance; #theta; Electronic noise [MeV]")
    h_elecNoise_withTraceCap[i].SetName("h_elecNoise_withTraceCap_"+str(i+1))

    h_elecNoise_shield.append( TH1F() )
    h_elecNoise_shield[i].SetLineWidth(3)
    h_elecNoise_shield[i].SetLineColor(line_color_number)
    h_elecNoise_shield[i].SetLineStyle(line_style_number)
    h_elecNoise_shield[i].SetBins(nbins, thetaMin, thetaMax)
    h_elecNoise_shield[i].SetTitle("Electronic noise - shields; #theta; Electronic noise [MeV]")
    h_elecNoise_shield[i].SetName("h_elecNoise_shield_"+str(i+1))

    h_elecNoise_trace.append( TH1F() )
    h_elecNoise_trace[i].SetLineWidth(3)
    h_elecNoise_trace[i].SetLineColor(line_color_number)
    h_elecNoise_trace[i].SetLineStyle(line_style_number)
    h_elecNoise_trace[i].SetBins(nbins, thetaMin, thetaMax)
    h_elecNoise_trace[i].SetTitle("Electronic noise -traces; #theta; Electronic noise [MeV]")
    h_elecNoise_trace[i].SetName("h_elecNoise_trace_"+str(i+1))

    h_elecNoise_detector.append( TH1F() )
    h_elecNoise_detector[i].SetLineWidth(3)
    h_elecNoise_detector[i].SetLineColor(line_color_number)
    h_elecNoise_detector[i].SetLineStyle(line_style_number)
    h_elecNoise_detector[i].SetBins(nbins, thetaMin, thetaMax)
    h_elecNoise_detector[i].SetTitle("Electronic noise - detector; #theta; Electronic noise [MeV]")
    h_elecNoise_detector[i].SetName("h_elecNoise_detector_"+str(i+1))

    #Total capacitance plot (shield + trace + detector)
    hCapTotal.append( TH1F() )
    hCapTotal[i].SetBins(nbins, thetaMin, thetaMax)
    hCapTotal[i].SetLineColor(line_color_number)
    hCapTotal[i].SetLineStyle(line_style_number)
    hCapTotal[i].SetLineWidth(3)
    hCapTotal[i].SetTitle("Total capacitance; #theta; Capacitance [pF]")
    hCapTotal[i].SetName("hCapacitance"+str(i))

    ref_charge_1mev = get_ref_charge(SFfcc[i])
    if i != 0:
        print(noise_charge_rms)
    print(i, " ", ref_charge_1mev)
    for ibin in range(0, nbins+1):
        capShield = hCapShield[i].GetBinContent(ibin)
        capTrace = hCapTrace[i].GetBinContent(ibin)
        capDetector = hCapDetector[i].GetBinContent(ibin)
        #total capacitance
        #hCapTotal[i].SetBinContent( ibin, capShield + capDetector )
        #JP Include transferline capacitance as confirmed by C de la T.
        hCapTotal[i].SetBinContent( ibin, capShield + capTrace + capDetector )

        # Compute the NOISE
        noise_charge_rms = get_noise_charge_rms(capShield + capDetector + capTrace) #JP trace added!
        noise = noise_charge_rms / (ref_charge_1mev)
        noiseWithTrace = get_noise_charge_rms(capShield + capDetector + capTrace) / ref_charge_1mev
        noiseShield = get_noise_charge_rms(capShield) / ref_charge_1mev
        noiseTrace = get_noise_charge_rms(capTrace) / ref_charge_1mev
        noiseDetector = get_noise_charge_rms(capDetector) / ref_charge_1mev
        #find maximum for drawing of histograms
        if noise > maximumNoise:
            maximumNoise = noise
        if noiseWithTrace > maximumNoiseWithTrace:
            maximumNoiseWithTrace = noiseWithTrace
        if (capShield + capDetector)>maximumCap:
            maximumCap = (capShield + capDetector)
            
        #fill histogram
        #default 2025: now trace capacitance added everywhere
        h_elecNoise_fcc[i].SetBinContent(ibin, noise)
        h_elecNoise_all[i].SetBinContent(ibin, noise)
        h_elecNoise_withTraceCap[i].SetBinContent(ibin, noiseWithTrace)
        h_elecNoise_shield[i].SetBinContent(ibin, noiseShield)
        h_elecNoise_trace[i].SetBinContent(ibin, noiseTrace)
        h_elecNoise_detector[i].SetBinContent(ibin, noiseDetector)
        if ibin==floor((nbins+1)/2):
            print("layer %d" %(i+1), "eta==0: capacitance %.0f pF," %( capShield + capTrace + capDetector ), "total elec. noise %.4f MeV" %noise, "elec. noise without trace cap. %.4f MeV" %noiseWithTrace)
    line_color_number += 1
    line_style_number += 1

#print maximumCap, maximumNoise, maximumNoiseWithTrace

cCapacitance = TCanvas("cCapacitance","Capacitance per cell",800,600)
cCapacitance.cd()
#legend = TLegend(0.135,0.573,0.466,0.872)
legend = TLegend(0.135,0.693,0.8,0.892)
legend.SetBorderSize(0)
legend.SetLineColor(0)
legend.SetLineStyle(0)
legend.SetLineWidth(0)
legend.SetFillColor(0)
legend.SetFillStyle(0)
legend.SetHeader("Longitudinal layers")
legend.SetNColumns(4)
for i, h in enumerate(hCapTotal):
    h.SetMinimum(0)
    h.SetMaximum(maximumCap*1.5)
    h.GetYaxis().SetTitleOffset(1.4)
    if i == 0:
        h.Draw("")
    else:
        h.Draw("same")
    legend.AddEntry(h,"Layer " + str(i+1),"l")


# Prepare "nice" plots & save all capacitances + noise
fSaveAll.cd()

for h in hCapTotal:
    h.SetMinimum(0.)
    #h.SetMaximum(maximumCap*1.3)
    h.GetYaxis().SetTitleOffset(1.4)
    h.Write()

legend.Draw()
cCapacitance.Update()
cCapacitance.Write()
cCapacitance.Print(os.path.join(output_folder, "cCapacitance.png"))

#maximumCap = 1200.
#maximumNoise = 0.04
#maximumNoiseWithTrace = 0.015

for h in itertools.chain(h_elecNoise_fcc, h_elecNoise_all):
    h.SetMinimum(0.)
    h.SetMaximum(maximumNoise*1.5)
    h.GetYaxis().SetTitleOffset(1.4)
    h.Write()

for h in itertools.chain(h_elecNoise_withTraceCap, h_elecNoise_shield, h_elecNoise_trace, h_elecNoise_detector):
    h.SetMinimum(0.)
    h.SetMaximum(maximumNoiseWithTrace*1.5)
    h.GetYaxis().SetTitleOffset(1.4)
    h.Write()

cNoise = TCanvas("cNoise","Electronic noise per cell",800,600)
cNoise.cd()
for i, h in enumerate(h_elecNoise_fcc):
    if i == 0:
        h.Draw("")
    else:
        h.Draw("same")

legend.Draw()
cNoise.Update()
cNoise.Write()
cNoise.Print(os.path.join(output_folder, "cNoise.png"))

cNoiseWithTrace = TCanvas("cNoiseWithTrace","Electronic noise with trace cap. per cell",800,600)
cNoiseWithTrace.cd()
for i, h in enumerate(h_elecNoise_withTraceCap):    
    if i == 0:
        h.Draw("")
    else:
        h.Draw("same")

legend.Draw()
cNoiseWithTrace.Update()
cNoiseWithTrace.Write()
cNoiseWithTrace.Print(os.path.join(output_folder, "cNoiseWithTrace.png"))

legendP = TLegend(0.1,0.6,0.43,0.9)
legendP.SetHeader("Capacitance")

cCapParts = TCanvas("cCapParts","",1200,1000)
cCapParts.Divide(3,4)    
for i in range (0, nLayers):
    cCapParts.cd(i+1)
    if i < 7:
        hCapTotal[i].SetMaximum(100)
    else:
        hCapTotal[i].SetMaximum(200)

    hCapTotal[i].SetTitle("Layer "+str(i+1))
    hCapTotal[i].GetXaxis().SetTitleSize(0.045)
    hCapTotal[i].GetYaxis().SetTitleSize(0.045)
    hCapTotal[i].GetYaxis().SetTitleOffset(1.15)
    hCapTotal[i].SetLineColor(1)
    hCapTotal[i].SetLineStyle(1)
    hCapShield[i].SetLineColor(2)
    hCapShield[i].SetLineStyle(2)
    hCapTrace[i].SetLineColor(3)
    hCapTrace[i].SetLineStyle(3)
    hCapDetector[i].SetLineColor(4)
    hCapDetector[i].SetLineStyle(4)
    hCapTotal[i].Draw()
    hCapShield[i].Draw("same")
    #hCapTrace[i].Draw("same")
    hCapDetector[i].Draw("same")
    gPad.Update()
    if i==0:
        legendP.AddEntry(hCapTotal[i],"total cap.","l")
        legendP.AddEntry(hCapShield[i],"shield cap.","l")
        #legendP.AddEntry(hCapTrace[i],"trace cap.","l")
        legendP.AddEntry(hCapDetector[i],"detector cap.","l")
    legendP.Draw()

cCapParts.Write()
cCapParts.Print(os.path.join(output_folder, "cCapParts.png"))

#Save final noise plot (to be used in FCCSW)
fSave.cd()
for h in h_elecNoise_fcc:
    h.Write()

#closeInput = raw_input("Press ENTER to exit")


