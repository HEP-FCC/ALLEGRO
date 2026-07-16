import ROOT
import math
import dd4hep
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("compactFile", type=str, help="Top-level compact file name")
args = parser.parse_args()

path_to_detector = os.environ.get("K4GEO", "")
detectorFile = path_to_detector + "/" + args.compactFile

# check that file exists
if not os.path.isfile(detectorFile):
    print(f"Error: compact file '{detectorFile}' does not exist.", file=sys.stderr)
    sys.exit(1)

detector = dd4hep.Detector.getInstance()
detector.fromXML(detectorFile)

nWheels = detector.constantAsLong("EMECnWheels")

EMECNumCalibRhoLayers = [detector.constantAsLong("EMECNumCalibRhoLayersWheel1"),
                detector.constantAsLong("EMECNumCalibRhoLayersWheel2"),
                detector.constantAsLong("EMECNumCalibRhoLayersWheel3")]

EMECNumCalibZLayers =  [detector.constantAsLong("EMECNumCalibZLayersWheel1"),
                detector.constantAsLong("EMECNumCalibZLayersWheel2"),
                detector.constantAsLong("EMECNumCalibZLayersWheel3")]
EMECNumReadoutRhoLayers =  [detector.constantAsLong("EMECNumReadoutRhoLayersWheel1"),
                detector.constantAsLong("EMECNumReadoutRhoLayersWheel2"),
                detector.constantAsLong("EMECNumReadoutRhoLayersWheel3")]
EMECNumReadoutZLayers = [detector.constantAsLong("EMECNumReadoutZLayersWheel1"),
                detector.constantAsLong("EMECNumReadoutZLayersWheel2"),
                detector.constantAsLong("EMECNumReadoutZLayersWheel3")]

nLayers = 0

for i in range(0, nWheels) :
    nLayers += EMECNumCalibRhoLayers[i]*EMECNumCalibZLayers[i]

SF = (
    [0.0897818] * 1
    + [0.221318] * 1
    + [0.0820002] * 1
    + [0.994281] * 1
    + [0.0414437] * 1
    + [0.1148] * 1
    + [0.178831] * 1
    + [0.142449] * 1
    + [0.181206] * 1
    + [0.342843] * 1
    + [0.137479] * 1
    + [0.176479] * 1
    + [0.153273] * 1
    + [0.195836] * 1
    + [0.0780405] * 1
    + [0.150202] * 1
    + [0.17846] * 1
    + [0.164886] * 1
    + [0.175758] * 1
    + [0.10836] * 1
    + [0.160243] * 1
    + [0.183373] * 1
    + [0.171818] * 1
    + [0.194848] * 1
    + [0.111899] * 1
    + [0.170704] * 1
    + [0.188455] * 1
    + [0.178164] * 1
    + [0.209113] * 1
    + [0.105241] * 1
    + [0.180637] * 1
    + [0.192206] * 1
    + [0.186096] * 1
    + [0.211962] * 1
    + [0.112019] * 1
    + [0.180344] * 1
    + [0.195684] * 1
    + [0.190778] * 1
    + [0.218259] * 1
    + [0.118516] * 1
    + [0.207786] * 1
    + [0.204474] * 1
    + [0.207048] * 1
    + [0.225913] * 1
    + [0.111325] * 1
    + [0.147875] * 1
    + [0.195625] * 1
    + [0.173326] * 1
    + [0.175449] * 1
    + [0.104087] * 1
    + [0.153645] * 1
    + [0.161263] * 1
    + [0.165499] * 1
    + [0.171758] * 1
    + [0.175789] * 1
    + [0.180657] * 1
    + [0.184563] * 1
    + [0.187876] * 1
    + [0.191762] * 1
    + [0.19426] * 1
    + [0.197959] * 1
    + [0.199021] * 1
    + [0.204428] * 1
    + [0.195709] * 1
    + [0.151751] * 1
    + [0.171477] * 1
    + [0.165509] * 1
    + [0.172565] * 1
    + [0.172961] * 1
    + [0.175534] * 1
    + [0.177989] * 1
    + [0.18026] * 1
    + [0.181898] * 1
    + [0.183912] * 1
    + [0.185654] * 1
    + [0.187515] * 1
    + [0.190408] * 1
    + [0.188794] * 1
    + [0.193699] * 1
    + [0.192287] * 1
    + [0.19755] * 1
    + [0.190943] * 1
    + [0.218553] * 1
    + [0.161085] * 1
    + [0.373086] * 1
    + [0.122495] * 1
    + [0.21103] * 1
    + [1] * 1
    + [0.138686] * 1
    + [0.0545171] * 1
    + [1] * 1
    + [1] * 1
    + [0.227945] * 1
    + [0.0122872] * 1
    + [0.00437334] * 1
    + [0.00363533] * 1
    + [1] * 1
    + [1] * 1
)

if (len(SF) != nLayers) :
    print ("Error: number of entries in sF list does not match number of layers")

def get_layer(iWheel,iRho, iZ):
    layerOffset = 0
    if (iWheel == 1) :
        layerOffset = EMECNumCalibRhoLayers[0]*EMECNumCalibZLayers[0]
    if (iWheel == 2):
        layerOffset = EMECNumCalibRhoLayers[0]*EMECNumCalibZLayers[0] +  EMECNumCalibRhoLayers[1]*EMECNumCalibZLayers[1]

    return int(layerOffset + iZ/(EMECNumReadoutZLayers[iWheel]/EMECNumCalibZLayers[iWheel]) + EMECNumCalibZLayers[iWheel]*(iRho/(EMECNumReadoutRhoLayers[iWheel]/EMECNumCalibRhoLayers[iWheel]) ))

#conversion from capacitance to noise in electrons, assuming cold electronics
# taken from Omega lab measurements as reported in https://indico.cern.ch/event/1545838/contributions/6831866/attachments/3194785/5686292/capa_and_noise_juska_v0.pdf
def get_noise_charge_rms(capacitance):
    A = 16.5
    B = 945
    return math.sqrt(A * capacitance**2 + B**2) # number of electrons

# Get the equivalent of 1 MeV energy deposit in a cell (absorber + Lar) in terms of number of electrons in the charge pre-amplifier + shaper
r_recomb = 0.04
w_lar = 23.6 # eV needed to create a ion/electron pair
def get_ref_charge(SF, E_dep = 1 * pow(10, 6)): #E_dep en eV, choose 1 MeV
    return E_dep * SF * (1 - r_recomb) / (2 * w_lar) # nA, the factor 2 comes from: Q_tot = I_0 * t_drift * 1/2  (rectangle --> triangle), t_drift cancels out from the formula to get I_0 which has v_drift/d_gap (Ramo Shockley).
                                                     # Assumption: shaping time is similar or bigger to drift time


filename = "endcap_capacitances.root"
fIn = ROOT.TFile(filename, "r")
hIn =fIn.Get("endcap_capacitances")

output_folder = "noise_capa_ecalendcap"
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

fSave = ROOT.TFile(os.path.join(output_folder, "elecNoise_ecalendcap.root"),"RECREATE")
h_elecNoise_fcc = []

for iWheel in range(0,nWheels) :
    hNoise = ROOT.TH2F("noise_endcap_wheel"+str(iWheel+1), "noise_endcap_wheel"+str(iWheel+1), EMECNumReadoutZLayers[iWheel], 0, EMECNumReadoutZLayers[iWheel], EMECNumReadoutRhoLayers[iWheel], 0, EMECNumReadoutRhoLayers[iWheel])
    for iZ in range (0, EMECNumReadoutZLayers[iWheel]) :
        for iRho in range (0, EMECNumReadoutRhoLayers[iWheel]) :
            layer = get_layer(iWheel, iRho, iZ)
            ref_charge_1mev = get_ref_charge(SF[layer])
            binIndex = hIn.GetBin(iWheel+1, iZ+1)
            cap = hIn.GetBinContent(binIndex)
            noise_RMS = get_noise_charge_rms(cap) / ref_charge_1mev
            hNoise.Fill(iZ, iRho, noise_RMS)

    h_elecNoise_fcc.append(hNoise)

fSave.cd()
for iHist in range (0, nWheels) :
    h_elecNoise_fcc[iHist].Write()

fSave.Close()


