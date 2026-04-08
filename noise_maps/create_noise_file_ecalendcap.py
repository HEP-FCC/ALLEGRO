import ROOT
import math
import os

#start with some numbers that should ideally be read from xml:
nWheels = 3
ECalEndcapNumCalibRhoLayers = [10, 14, 34]
ECalEndcapNumCalibZLayers = [5,1,1]
ECalEndcapNumReadoutRhoLayers = [10, 14, 34]
ECalEndcapNumReadoutZLayers = [10,10,10]
# end of numbers from xml

nLayers = 0

for i in range(0, nWheels) :
    nLayers += ECalEndcapNumCalibRhoLayers[i]*ECalEndcapNumCalibZLayers[i]

SF = [0.0977728] * 1+ [0.124049] * 1+ [0.0826009] * 1+ [0.185245] * 1+ [0.0438168] * 1+ [0.118106] * 1+ [0.126966] * 1+ [0.124955] * 1+ [0.120205] * 1+ [0.376882] * 1+ [0.134537] * 1+ [0.140725] * 1+ [0.141929] * 1+ [0.139637] * 1+ [0.0961235] * 1+ [0.148974] * 1+ [0.155324] * 1+ [0.156519] * 1+ [0.159014] * 1+ [0.106143] * 1+ [0.162649] * 1+ [0.168907] * 1+ [0.170696] * 1+ [0.170492] * 1+ [0.115022] * 1+ [0.17437] * 1+ [0.182934] * 1+ [0.181427] * 1+ [0.193245] * 1+ [0.115148] * 1+ [0.191578] * 1+ [0.19504] * 1+ [0.19817] * 1+ [0.214383] * 1+ [0.129678] * 1+ [0.189389] * 1+ [0.207888] * 1+ [0.206941] * 1+ [0.245014] * 1+ [0.105342] * 1+ [0.237301] * 1+ [0.231047] * 1+ [0.240688] * 1+ [0.247344] * 1+ [0.125327] * 1+ [0.160006] * 1+ [0.224148] * 1+ [0.204044] * 1+ [0.186107] * 1+ [0.129094] * 1+ [0.107181] * 1+ [0.116029] * 1+ [0.127742] * 1+ [0.139121] * 1+ [0.150254] * 1+ [0.16093] * 1+ [0.171617] * 1+ [0.181933] * 1+ [0.191871] * 1+ [0.200919] * 1+ [0.212304] * 1+ [0.217326] * 1+ [0.236451] * 1+ [0.222528] * 1+ [0.102471] * 1+ [0.111046] * 1+ [0.113203] * 1+ [0.119741] * 1+ [0.123478] * 1+ [0.12913] * 1+ [0.133156] * 1+ [0.138544] * 1+ [0.142485] * 1+ [0.147565] * 1+ [0.151996] * 1+ [0.15606] * 1+ [0.160749] * 1+ [0.165321] * 1+ [0.169641] * 1+ [0.173081] * 1+ [0.179026] * 1+ [0.181076] * 1+ [0.18676] * 1+ [0.190037] * 1+ [0.195085] * 1+ [0.196907] * 1+ [0.204426] * 1+ [0.204138] * 1+ [0.211934] * 1+ [0.213463] * 1+ [0.216201] * 1+ [0.223871] * 1+ [0.21637] * 1+ [0.255613] * 1+ [0.243175] * 1+ [0.108932] * 1+ [-0.0752322] * 1+ [0.0276576] * 1

if (len(SF) != nLayers) :
    print ("Error: number of entries in sF list does not match number of layers")

def get_layer(iWheel,iRho, iZ):
    layerOffset = 0
    if (iWheel == 1) :
        layerOffset = ECalEndcapNumCalibRhoLayers[0]*ECalEndcapNumCalibZLayers[0]
    if (iWheel == 2):
        layerOffset = ECalEndcapNumCalibRhoLayers[0]*ECalEndcapNumCalibZLayers[0] +  ECalEndcapNumCalibRhoLayers[1]*ECalEndcapNumCalibZLayers[1]
        
    return int(layerOffset + iZ/(ECalEndcapNumReadoutZLayers[iWheel]/ECalEndcapNumCalibZLayers[iWheel]) + ECalEndcapNumCalibZLayers[iWheel]*(iRho/(ECalEndcapNumReadoutRhoLayers[iWheel]/ECalEndcapNumCalibRhoLayers[iWheel]) ))

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
    hNoise = ROOT.TH2F("noise_endcap_wheel"+str(iWheel), "noise_endcap_wheel"+str(iWheel), ECalEndcapNumReadoutZLayers[iWheel], 0, ECalEndcapNumReadoutZLayers[iWheel], ECalEndcapNumReadoutRhoLayers[iWheel], 0, ECalEndcapNumReadoutRhoLayers[iWheel])
    for iZ in range (0, ECalEndcapNumReadoutZLayers[iWheel]) :
        for iRho in range (0, ECalEndcapNumReadoutRhoLayers[iWheel]) :
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
    
            
