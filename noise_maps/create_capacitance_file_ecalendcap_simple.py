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
detector = dd4hep.Detector.getInstance()
detector.fromXML(detectorFile)

n_wheels = detector.constantAsLong("EMECnWheels")
blade_angles = [ detector.constantAsDouble("EMECBladeAngle1"),
                 detector.constantAsDouble("EMECBladeAngle2"),
                 detector.constantAsDouble("EMECBladeAngle3")]

n_cells_in_z = [detector.constantAsLong("EMECNumReadoutZLayersWheel1"),
                detector.constantAsLong("EMECNumReadoutZLayersWheel2"),
                detector.constantAsLong("EMECNumReadoutZLayersWheel3")]
                
z_depth = detector.constantAsDouble("EMEC_z2") - detector.constantAsDouble("EMEC_z1")


strip_C_per_cm = 1.9 # pF per cm, taken from barrel prototype readout board measurements

shield_to_pad_C_per_cm = 1.2 # pF per cm, taken from barrel prototype readout board measurements

f_out = ROOT.TFile("endcap_capacitances.root", "RECREATE")

h_cap = ROOT.TH2F("endcap_capacitances", "Endcap capacitances", n_wheels, 0, n_wheels, max(n_cells_in_z), 0, max(n_cells_in_z));

for iWheel in range(0,n_wheels):
    cell_size = z_depth/n_cells_in_z[iWheel]/math.cos(blade_angles[iWheel])
    for iZ in range (0, n_cells_in_z[iWheel]) :
        dist = cell_size/2 + (n_cells_in_z[iWheel]-iZ-1)*cell_size
        C = dist*(strip_C_per_cm + strip_C_per_cm )
        h_cap.Fill(iWheel, iZ, C)

f_out.cd()
h_cap.Write()
f_out.Close()
 


 
        
    
