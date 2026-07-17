import os
import sys
from xml.dom import minidom
import re
from math import radians, sqrt, cos
# python write_calibration_xml.py ../../k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ECalBarrel_thetamodulemerged.xml

def add_newline(xmlfile):
    with open(xmlfile, 'a') as file:
        file.write("\n")

def get_length(node):
    string = node.getAttribute('value')
    if not string.split("*")[1] == 'mm':
        print("Error in orignal xml file, expected length in mm, exiting...")
        sys.exit(1)
    return float(string.split("*")[0])


input_xml_path = sys.argv[1]
output_xml_path_sf = input_xml_path.replace(".xml", "_calibration.xml")
output_xml_path_upstream = input_xml_path.replace(".xml", "_upstream.xml")
detDim_xml_path = os.path.join(os.path.dirname(input_xml_path), "DectDimensions.xml")

list_of_pair_layerThickness_numberOfLayer = []

input_xml = minidom.parse(input_xml_path)
# print input_xml.toprettyxml()

numberOfLayer = 0
n_phi_bins = 0
n_modules = 0
eta_bin_size = 0
eta_bin_size_eval = 0
theta_bin_size = 0
theta_bin_size_eval = 0
original_cryo_back_size = 0
# print input_xml.getElementsByTagName('lccdd')

# First find the detector dimension
detDim_xml = minidom.parse(detDim_xml_path)
BarECal_rmin = '0'
BarECal_rmax = '0'
for nodeList in detDim_xml.getElementsByTagName('lccdd'):
    for node in nodeList.childNodes:
        if node.localName == 'define':
            for subnode in node.childNodes:
                if subnode.localName == 'constant' and subnode.getAttribute('name') == 'BarECal_rmin':
                    BarECal_rmin = get_length(subnode)
                if subnode.localName == 'constant' and subnode.getAttribute('name') == 'BarECal_rmax':
                    BarECal_rmax = get_length(subnode)

print("rin =", BarECal_rmin)
print("rout =", BarECal_rmax)

cryoBarrelFrontWarm = 0.0
cryoBarrelFrontCold = 0.0
nliqBathThicknessFront = 0.0
airMarginThickness = 0.0
safetyMargin = 0.0
alpha = 0.0
for nodeList in input_xml.getElementsByTagName('lccdd'):
    for node in nodeList.childNodes:
        # get the cryostat back size
        if node.localName == 'define':
            for subnode in node.childNodes:
                if subnode.localName == 'constant':
                    if subnode.getAttribute('name') == 'CryoBarrelFrontWarm':
                        cryoBarrelFrontWarm = get_length(subnode)
                    elif subnode.getAttribute('name') == 'CryoBarrelFrontCold':
                        cryoBarrelFrontCold = get_length(subnode)
                    elif subnode.getAttribute('name') == 'NLiqBathThicknessFront':
                        nliqBathThicknessFront = get_length(subnode)
                    elif subnode.getAttribute('name') == 'AirMarginThickness':
                        airMarginThickness = get_length(subnode)
                    elif subnode.getAttribute('name') == 'SafetyMargin':
                        safetyMargin = get_length(subnode)
print("Cryo barrel front cold =", cryoBarrelFrontCold)
print("Cryo barrel front warm =", cryoBarrelFrontWarm)
print("Air margin thickness =", airMarginThickness)
print("NLiq bath thickness =", nliqBathThicknessFront)
print("Safety margin =", safetyMargin)
bath_rmin = BarECal_rmin + airMarginThickness + cryoBarrelFrontCold + cryoBarrelFrontWarm + nliqBathThicknessFront - safetyMargin
print("Bath rmin =", bath_rmin)

# write xml file for calculation of sampling fractions, turning passive material into active
for nodeList in input_xml.getElementsByTagName('lccdd'):
    for node in nodeList.childNodes:
        # get the cryostat back size
        if node.localName == 'define':
            for subnode in node.childNodes:
                if subnode.localName == 'constant' and subnode.getAttribute('name') == 'CryoBarrelBackCold':
                    original_cryo_back_size_str = subnode.getAttribute('value')
                    if not original_cryo_back_size_str.split("*")[1] == 'mm':
                        print("Error in orignal xml file, cryo thickness expected in mm, exiting...")
                        sys.exit(1)
                    original_cryo_back_size = int(original_cryo_back_size_str.split("*")[0])
                if subnode.localName == 'constant' and subnode.getAttribute('name') == 'ECalBarrelNumPlanes':
                    n_modules = int(subnode.getAttribute('value'))
                if subnode.localName == 'constant' and subnode.getAttribute('name') == 'InclinationAngle':
                    alpha  = subnode.getAttribute('value').strip()
                    m = re.fullmatch(r'([+-]?\d*\.?\d+)\s*(?:\*\s*degree)?', alpha)
                    alpha = radians(float(m.group(1))) if "*" in alpha else float(m.group(1))
                    print("Inclination angle (radians) =", alpha)
        if node.localName == 'detectors':
            for subnode in node.childNodes:
                if subnode.localName == 'detector':
                    print(subnode.localName)
                    for subsubnode in subnode.childNodes:
                        if subsubnode.localName == 'calorimeter':
                            print("    ", subsubnode.localName)
                            for subsubsubnode in subsubnode.childNodes:
                                if subsubsubnode.localName == 'readout':
                                    print("        ", subsubsubnode.localName)
                                    # print subsubsubnode.getAttribute('sensitive')
                                    subsubsubnode.setAttribute('sensitive', 'true')  # here we change the readout as sensitive
                                    # print subsubsubnode.getAttribute('sensitive')
                                if subsubsubnode.localName == 'passive':
                                    print("        ", subsubsubnode.localName)
                                    for subsubsubsubnode in subsubsubnode.childNodes:
                                        if subsubsubsubnode.localName in ['inner', 'innerMax', 'glue', 'outer']:
                                            print("            ", subsubsubsubnode.localName)
                                            # print subsubsubsubnode.getAttribute('sensitive')
                                            subsubsubsubnode.setAttribute('sensitive', 'true')  # here we change the absorber into sensitive material
                                            # print subsubsubsubnode.getAttribute('sensitive')
                                if subsubsubnode.localName == 'layers':
                                    for subsubsubsubnode in subsubsubnode.childNodes:
                                        if subsubsubsubnode.localName == 'layer':
                                            numberOfLayer += int(subsubsubsubnode.getAttribute('repeat'))
                                            list_of_pair_layerThickness_numberOfLayer.append([subsubsubsubnode.getAttribute('thickness').split('*')[0], subsubsubsubnode.getAttribute('repeat')])
        if node.localName == 'readouts':
            for subnode in node.childNodes:
                if subnode.localName == 'readout' and subnode.getAttribute('name') == 'ECalBarrelPhiEta':
                    for subsubnode in subnode.childNodes:
                        if subsubnode.localName == 'segmentation':
                            n_phi_bins = int(subsubnode.getAttribute('phi_bins'))
                            eta_bin_size = subsubnode.getAttribute('grid_size_eta')
                            eta_bin_size_eval = eval(eta_bin_size)
                if subnode.localName == 'readout' and subnode.getAttribute('name') == 'ECalBarrelModuleThetaMerged':
                    for subsubnode in subnode.childNodes:
                        if subsubnode.localName == 'segmentation':
                            # n_modules = subsubnode.getAttribute('nModules')
                            theta_bin_size = subsubnode.getAttribute('grid_size_theta')
                            theta_bin_size_eval = eval(theta_bin_size)

with open(output_xml_path_sf, "w") as f:
    input_xml.writexml(f)
add_newline(output_xml_path_sf)
print(output_xml_path_sf, " written.")


# while parsing the input xml we have found out number and length along electrode (in cm) of layers, and theta/eta grid size and number of phi bins/modules
# so we update the .py files accordingly
# first, we calculate radial depth of layers (in cm)
r_layer_in = bath_rmin / 10.
radialDepths = []
layer = 0
string_for_layerWidth = ""
print("Number of layers: %d" % numberOfLayer)
print("Layer layout {length : number}: ", list_of_pair_layerThickness_numberOfLayer)
for pair_layerThickness_numberOfLayer in list_of_pair_layerThickness_numberOfLayer:
    nLayers = int(pair_layerThickness_numberOfLayer[1])
    layerLength = float(pair_layerThickness_numberOfLayer[0])
    for iLayer in range(nLayers):
        print(f"Layer {layer}: length along electrode = {layerLength} cm")
        r_layer_out = sqrt(layerLength*layerLength + r_layer_in*r_layer_in + 2*layerLength*r_layer_in*cos(alpha))
        radialDepths.append(r_layer_out - r_layer_in)
        r_layer_in = r_layer_out
        if layer == 0:
            string_for_layerWidth += "[%f] * 1" % radialDepths[layer]
        else:
            string_for_layerWidth += " + [%f] * 1" % radialDepths[layer]
        layer += 1
print("Radial depths: ",radialDepths)

# modify the number of layer in sampling fraction and upstream config files
os.system("sed -i 's/ecalBarrelLayers =.*/ecalBarrelLayers = %d/' $ALLEGRO/sampling_fractions/fcc_ee_samplingFraction_inclinedEcal.py $ALLEGRO/upstream/fcc_ee_upstream_inclinedEcal.py" % numberOfLayer)
print("number of layers updated in $ALLEGRO/sampling_fractions/fcc_ee_samplingFraction_inclinedEcal.py and $ALLEGRO/upstream/fcc_ee_upstream_inclinedEcal.py")

# modify the layer layout in plot_sampling_fraction script
os.system(r"sed -i 's/default=\[1\] \*.*,/default=\[1\] \* %d,/' $ALLEGRO/sampling_fractions/FCC_calo_analysis_cpp/plot_samplingFraction.py" % numberOfLayer)
os.system("sed -i 's/totalNumLayers\", default = .*,/totalNumLayers\", default = %d,/' $ALLEGRO/sampling_fractions/FCC_calo_analysis_cpp/plot_samplingFraction.py" % numberOfLayer)
print("number of layers updated in $ALLEGRO/sampling_fractions/FCC_calo_analysis_cpp/plot_samplingFraction.py")
os.system("sed -i 's/layerWidth\", default=.*,/layerWidth\", default=%s,/' $ALLEGRO/sampling_fractions/FCC_calo_analysis_cpp/plot_samplingFraction.py" % string_for_layerWidth)
print("number and thickness of layers updated in $ALLEGRO/sampling_fractions/FCC_calo_analysis_cpp/plot_samplingFraction.py")

# modify the number of layers in run_ALLEGRO_reco.py
os.system("sed -i --follow-symlinks 's/ecalBarrelLayers = .*/ecalBarrelLayers = %d/' $FCCBASEDIR/run/run_ALLEGRO_reco.py" % numberOfLayer)
print("number of layers updated in run_ALLEGRO_reco.py")

# modify the number of layers in noise_map.py and neighbours.py in ALLEGRO
os.system("sed -i 's/ecalBarrelNumLayers = .*/ecalBarrelNumLayers = %d/' $ALLEGRO/noise_maps/noise_map.py $ALLEGRO/neighbor_maps/neighbours.py" % numberOfLayer)
print("number of layers updated in $ALLEGRO/noise_map.py and $ALLEGRO/neighbor_maps/neighbours.py")

# modify the number of layers in noise_map.py and neighbours.py in ALLEGRO
os.system("sed -i 's/ecalBarrelNumLayers = .*/ecalBarrelNumLayers = %d/' $ALLEGRO/noise_maps/noise_map.py $ALLEGRO/neighbor_maps/neighbours.py" % numberOfLayer)
print("number of layers updated in $ALLEGRO/noise_maps/noise_map.py and $ALLEGRO/neighbor_maps/neighbours.py")

# modify the tower definition in clustering algorithms
# the *4 for theta is because the grid size reflects the width of the strip cells, but the
# tower sizes are defined in units of big cells (corresponding to 4 cells)
if theta_bin_size_eval > 0 and n_modules > 0:
    os.system(r"sed -i --follow-symlinks 's#deltaThetaTower.*$#deltaThetaTower=4 * %s, deltaPhiTower=2 * 2 * pi \/ %d.,#'  $FCCBASEDIR/run/run_ALLEGRO_reco.py" % (theta_bin_size.replace("/", " / "), n_modules))
    print("theta-phi tower size updated in run_ALLEGRO_reco.py")


# Write upstream correction xml
# Re-make absorber and readout not sensitive, make cryostat sensitive and enlarge it to 1.1 m
new_cryo_back_size = 1100  # mm
BarECal_rmax_int = int(BarECal_rmax)
new_BarECal_rmax_int = BarECal_rmax_int + (new_cryo_back_size - original_cryo_back_size)
print("New rout =", new_BarECal_rmax_int)

added_rmax = False
for nodeList in input_xml.getElementsByTagName('lccdd'):
    for node in nodeList.childNodes:
        if node.localName == 'define':
            for subnode in node.childNodes:
                if not added_rmax:
                    rMaxNode = input_xml.createElement('constant')
                    rMaxNode.setAttribute('name', 'BarECal_rmax')
                    rMaxNode.setAttribute('value', '%d*mm' % new_BarECal_rmax_int)
                    node.insertBefore(rMaxNode, subnode)
                    added_rmax = True
                if subnode.localName == 'constant' and subnode.getAttribute('name') == "CryoBarrelBackCold":
                    subnode.setAttribute('value', "%d*mm" % new_cryo_back_size)
        if node.localName == 'detectors':
            for subnode in node.childNodes:
                if subnode.localName == 'detector':
                    for subsubnode in subnode.childNodes:
                        if subsubnode.localName == 'calorimeter':
                            for subsubsubnode in subsubnode.childNodes:
                                if subsubsubnode.localName == 'readout':
                                    subsubsubnode.setAttribute('sensitive', 'false')
                                if subsubsubnode.localName == 'passive':
                                    for subsubsubsubnode in subsubsubnode.childNodes:
                                        if subsubsubsubnode.localName in ['inner', 'innerMax', 'glue', 'outer']:
                                            subsubsubsubnode.setAttribute('sensitive', 'false')
                        if subsubnode.localName == 'cryostat':
                            for subsubsubnode in subsubnode.childNodes:
                                if subsubsubnode.localName in ['front', 'side', 'back']:
                                    subsubsubnode.setAttribute('sensitive', 'true')  # here we change the cryo front/side/back into sensitive material!

with open(output_xml_path_upstream, "w") as f:
    input_xml.writexml(f)
add_newline(output_xml_path_upstream)
print(output_xml_path_upstream, " written.")
