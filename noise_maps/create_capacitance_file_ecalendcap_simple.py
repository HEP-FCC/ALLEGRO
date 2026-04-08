import ROOT
import math

strip_C_per_cm = 1.9 # pF per cm, taken from barrel prototype readout board measurements

shield_to_pad_C_per_cm = 1.2 # pF per cm, taken from barrel prototype readout board measurements

z_depth = 45. # depth of the the endcap calorimeter, in cm.  Ideally would be read in from xml

blade_angle = 52. # degrees, ideally would be read in from xml
blade_angle = blade_angle*math.pi/180

n_cells_in_z = [10, 10, 10] # ideally would be read in from xml

n_wheels = 3  # ideally would be read in from XML


f_out = ROOT.TFile("endcap_capacitances.root", "RECREATE")

h_cap = ROOT.TH2F("endcap_capacitances", "Endcap capacitances", n_wheels, 0, n_wheels, max(n_cells_in_z), 0, max(n_cells_in_z));

for iWheel in range(0,n_wheels):
    cell_size = z_depth/n_cells_in_z[iWheel]/math.cos(blade_angle)
    for iZ in range (0, n_cells_in_z[iWheel]) :
        dist = cell_size/2 + (n_cells_in_z[iWheel]-iZ-1)*cell_size
        C = dist*(strip_C_per_cm + strip_C_per_cm )
        h_cap.Fill(iWheel, iZ, C)

f_out.cd()
h_cap.Write()
f_out.Close()
 


 
        
    
