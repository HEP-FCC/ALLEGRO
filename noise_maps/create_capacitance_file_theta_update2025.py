# calculate and save histograms of capacitance per source vs theta
# output is saved in ROOT file with given filename
#
# execute script with
# python create_capacitance_file_theta_CERN_PCBv2_fix_capadensities.py
#
# Updated from the previous verion by Juska in November 2025
#
# This is the second updated version where capacitance lenght densities are taken from measurements,
# shields-per-pad count is fixed and (soon) dielectric between signal pad and absorber
# is taken into account in the detector capacitance calculation


from ROOT import TH1F, TF1, TF2, TCanvas, TLegend, TFile, gStyle
import ROOT
from math import ceil, sin, cos, log, tan, pi, sqrt, asin, degrees

ROOT.gROOT.SetBatch(ROOT.kTRUE)

gStyle.SetPadTickY(1)

debug = False
verbose = True

# Add appendix to filenames for not mixing different versions

apdx = "_update2025"

# output file
filename = "capacitances_perSource_ecalBarrelFCCee_theta%s.root" % apdx

# layer 2 require special care as it is separated in several cells and that the shield run beneath the etch: cell 2 signal pad top capa: 0.68 + 0.20 = 0.88, cell 2 signal pad bot: 0.56 + 0.21 = 0.77, cell 3: 0.34 + 2.4 = 2.74, cell 4: 1 + 0.25 = 1.25, cell 5: 1.85 + 0.28 = 2.13

# layer containing the strips
stripLayer = 1 #JP So we start counting from zero here. Let's keep that in mind.
               # I keep the strips in the "1st" layer as it is still the baseline choice,
               # and the impact to capacitance map is small

# Dimensions #NOTE JP the below debug prints are before the 2025 update

# ALLEGRO v03, projective cell corners in phi, LAr gap size adjusted to obtain 1536 modules
# Number of modules read from detector metadata and used in readout class: 1536
# Number of layers read from detector metadata and used in readout class: 11
#  Debug: Number of layers: 11 total thickness 40.54
#  Info: ECAL cryostat: front: rmin (cm) = 214.9 rmax (cm) = 216.28 dz (cm) = 310
#  Info: ECAL cryostat: back: rmin (cm) = 261.83 rmax (cm) = 272.1 dz (cm) = 310
#  Info: ECAL cryostat: side: rmin (cm) = 216.28 rmax (cm) = 261.83 dz (cm) = 3.38
#  Info: ECAL services: front: rmin (cm) = 216.28 rmax (cm) = 216.98 dz (cm) = 306.62
#  Info: ECAL services: back: rmin (cm) = 258.13 rmax (cm) = 261.83 dz (cm) = 306.62
#  Info: ECAL bath: material = LAr rmin (cm) =  216.98 rmax (cm) = 258.13 thickness in front of ECal (cm) = 1 thickness behind ECal (cm) = 4
#  Info: ECAL calorimeter volume rmin (cm) =  217.28 rmax (cm) = 257.83
#  Info: passive inner material = Lead
#  and outer material = lArCaloSteel
#  thickness of inner part at inner radius (cm) =  0.18
#  thickness of inner part at outer radius (cm) =  0.18
#  thickness of outer part (cm) =  0.01
#  thickness of total (cm) =  0.2
#  rotation angle = 0.875806
#  Info: number of passive plates = 1536 azim. angle difference =  0.00409062
#  Info:  distance at inner radius (cm) = 0.888809
#  distance at outer radius (cm) = 1.05468
#  Info: readout material = PCB
#  thickness of readout planes (cm) =  0.12
#  number of readout layers = 11
#  Info: thickness of calorimeter (cm) = 40.55
#  length of passive or readout planes (cm) =  57.3937
#  Debug: Thickness of layer 0 : 2.33596
#  Debug: Thickness of layer 1 : 4.75685
#  Debug: Thickness of layer 2 : 4.89843
#  Debug: Thickness of layer 3 : 5.04
#  Debug: Thickness of layer 4 : 5.20989
#  Debug: Thickness of layer 5 : 5.36562
#  Debug: Thickness of layer 6 : 5.54966
#  Debug: Thickness of layer 7 : 5.73371
#  Debug: Thickness of layer 8 : 5.94607
#  Debug: Thickness of layer 9 : 6.15843
#  Debug: Thickness of layer 10 : 6.3991
#  Info: active material = LAr active layers thickness at inner radius (cm) = 0.249173 thickness at outer radious (cm) = 0.484647 making 94.502 % increase.
#  Info: active passive initial overlap (before subtraction) (cm) = 0.1 = 50 %

# Detector
rmin = 2172.8
Nplanes = 1536
inclination_degree = 50.18
angle = inclination_degree / 180. * pi  # inclination angle in radians
passiveThickness = 2.0  # mm
activeTotal = 405.5
inclinedTotal = 573.937
# careful, this is not really the radial spacing, it is, after dilution, the spacing in the parallel direction --> radial depth spacing will not be constant
readoutLayerRadialLengths = [1.69, 3.53, 3.69, 3.76, 3.84, 3.99, 4.15, 4.30, 4.45, 4.61, 4.45] # Updated to give real PCBv2 cell lengths

# [1.65, 3.36, 3.46, 3.56, 3.68, 3.79, 3.92, 4.05, 4.20, 4.35, 4.52] #JP These were the values before 2025 update

#JP PCBv2 cell lenghts by Juska: ls = [22, 46, 48, 48, 48, 48, 49, 50, 52, 54, 56, 58, 60, 58] (in mm)
#Re-calculated with radlen = l*sin(inclination_degree)
# It gives [1.69, 3.53, 3.69, 3.76, 3.84, 3.99, 4.15, 4.30, 4.45, 4.61, 4.45]

numLayers = len(readoutLayerRadialLengths)
# Segmentation
deltaTheta = 0.009817477 / 4
minTheta = 0.58905
maxTheta = pi - minTheta
numTheta = int(ceil((maxTheta - minTheta) / deltaTheta))
nMergedThetaCells = [4]*numLayers
nMergedThetaCells[stripLayer] = 1
nMergedModules = [2]*numLayers

# only one trace for strip layer because 4 cells instead of one. Version where we extract all channels from the back
tracesPerLayer = [i for i in range(numLayers)]
for i in range(stripLayer+1, numLayers):
    tracesPerLayer[i] += 3 # 4 -> 3 to get it right
    
#JP orig: traces per layer = [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#JP fixed:    
    
#JP restore strip layer trace count to zero - we will route the trace(s) between strips!
#tracesPerLayer[stripLayer] = 0 #FIXME this screwed up one trick so had to disable. Strip capa becomes correct due to "capa density" set to zero.

# PCB dimensions [mm] #JP Updated to CERN PCBv2. Old were mainly from Paris PCBv3.
hhv = 0.1
hs = 0.15
t = 0.035
tsh = 0.0175 #JP shields are thinner in PCBv2, needed to add this
w = 0.09
ws = 0.180
hm = 0.3
pcbThickness = 5 * t + 2 * tsh + 2 * hhv + 2 * hs + 2 * hm  # mm #JP updated also this equation
                                                                 # Resulting 1.31 mm matches measurement well

if verbose:
    print("minTheta =", minTheta)
    print("maxTheta =", maxTheta)
    print("numTheta =", numTheta)
    print("Nplanes =", Nplanes)
    print("rmin = %f mm" % rmin)
    print("activeTotal = %f mm" % activeTotal)
    print("readoutLayerRadialLengths (in cm) =", readoutLayerRadialLengths)
    print("inclination (deg) =", inclination_degree)
    print("inclinedTotal = %f mm" % inclinedTotal)
    print("passiveThickness = %f mm" % passiveThickness)
    print("number of layers =", numLayers)
    print("merged cells in theta =", nMergedThetaCells)
    print("merged modules =", nMergedModules)
    print("traces per layer =", tracesPerLayer)
    print("pcbThickness: %f mm" % pcbThickness)

# constants:
# distance from signal trace to shield (HS) - from impedance vs. trace width vs. distance to ground layer 2D plot (Z = 50 Ohm)
# trace width (W) - min value
# trace thickness (T) - min value
# distance from shield to the edge of PCB
# http://www.analog.com/media/en/training-seminars/design-handbooks/Basic-Linear-Design/Chapter12.pdf, page 40
# signal trace
epsilonR = 4.8  #JP in CERN PCBs we have actually 4.8, not 4.4
# conversion factor: 1 inch = 25.4 mm
inch2mm = 25.4

# capa per length from maxwel1 (pF/mm) OBSOLETE
# strip layer has smaller capacitance due to traces running beneath the anti-etch
capa_per_mm = [0.1149*2] * numLayers #JP Updated to value measured from PCBv2 T3 cell 14
                                     #JP I also double these values as these are not doubled later on as should
                                     # (because we have two signal-pad-plus-shields layers in one cell)
#JP I don't know if the "traces running beneath anti-etch" effect makes sense
# I put it to zero. There's zero shields running under signal pads of strips, except for
# one in the case of logical trace ordering, but we won't do that.

capa_per_mm[stripLayer] = 0# 0.0575*2 #JP See justification above

stripLineCapaDensity = 0.1868 # Measured from PCBv2 T3 cell 1

if verbose:
    print("capa_per_mm (pF/mm) = " , capa_per_mm)

# multiplicative factors
# for the trace, factor 2 because we have two HV plate / absorber capa per cell
nmultTrace = 2 #FIXME JP I already doubled the capa density but still now get a correct value. To be understood.
# for the shield, where we use maxwell, the extra factor 2 (two signal pad / shield capa) is already accounted for
nmultShield = 1
# dielectric constants
epsilonRLAr = 1.5  # LAr at 88 K #TODO Double-check from Martin's ATLAS source
epsilon0 = 8.854 / 1000.  # pF/mm


# Fill the layer length, trace length, etc
readoutLayerParallelLengths = []
real_radial_separation = [rmin]
real_radial_depth = []
inclinations_wrt_radial_dir_at_middleRadialDepth = []
trace_length = []
numLayers = len(readoutLayerRadialLengths)


dilution_factor = inclinedTotal / activeTotal # Dilution factor takes care of capa decrease due to gap widening (right?)
trace_length_inner = 0
trace_length_outer = 0

outer = False
current_electrode_length = 0

for idx in range(numLayers):  # first pass to get all length parallel to the readout, real radial separation, inclination at the middle of the layer

    readoutLayerRadialLengths[idx] *= 10 # change from cm to mm
    
    parallel_length = readoutLayerRadialLengths[idx] * dilution_factor
    
    # Tricky point: in the xml geo, you define 'radial'segmentation, but these depths will be the one parallel to the plates after scaling by the dilution factor --> even when setting constant radial depth, the geometry builder will make constant parallel length step, not constant radial steps
    
    readoutLayerParallelLengths.append(parallel_length)
    if outer:  # prepare the starting trace length when starting to extract by the back of the PCB
        trace_length_outer += parallel_length
    if tracesPerLayer[idx] == 0 and tracesPerLayer[idx - 1] == 0:
        outer = True
    # sqrt(r**2+(L1+i*L2)**2+2*r*(L1+i*L2)*cos(alpha)) where L1 = 2.68, L2=12.09, r=192, alpha=50)
    current_electrode_length += parallel_length
    real_radial_separation.append(sqrt(rmin * rmin + current_electrode_length * current_electrode_length + 2 * rmin * current_electrode_length * cos(angle)))
    real_radial_depth.append(real_radial_separation[idx + 1] - real_radial_separation[idx])
    # treating the fact that radial angle decreases when radial depth increase
    # angle comprise by lines from  1) Interaction point to inner right edge of a cell, 2) Interaction point to outer left edge of the considered cell (useful to get the plate angle with radial direction that changes with increasing R)
    # based on scalene triangle sine law A/sin(a) = B/sin(b) = C/sin(c) (outer left edge aligned on the Y axis)
    inclinations_wrt_radial_dir_at_middleRadialDepth.append(asin(rmin * sin(angle) / (real_radial_separation[idx] + ((real_radial_separation[idx + 1] - real_radial_separation[idx]) / 2))))


# second pass to get trace lengths
outer = False
for idx in range(numLayers):
    if tracesPerLayer[idx] == 0 and tracesPerLayer[idx - 1] == 0:  # we change direction
        outer = True
    if outer:
        trace_length.append(trace_length_outer)
        if idx == numLayers - 1:
            trace_length_outer == 0
            continue
        trace_length_outer -= readoutLayerParallelLengths[idx + 1]
    else:
        trace_length.append(trace_length_inner)
        trace_length_inner += readoutLayerParallelLengths[idx]
        
#JP The signal trace lengths are now the wrong way around in the array.
# (this did not have impact when transferline capa was neglected)
# Let's invert it and it should be fine for the capacitance calculation.
trace_length.reverse()

print('Readout radial lengths originally asked: ', readoutLayerRadialLengths)
print('Readout parallel lengths: ', readoutLayerParallelLengths)
print("Real radial separation: ", real_radial_separation)
print("Real radial depth: ", real_radial_depth)
print("inclinations_wrt_radial_dir_at_middleRadialDepth: ", [degrees(inclinations) for inclinations in inclinations_wrt_radial_dir_at_middleRadialDepth])
print("Signal trace length per layer: ", trace_length)


gStyle.SetOptStat(0)

cImpedance = TCanvas("cImpedance", "", 600, 800)
cImpedance.Divide(1, 2)
cImpedance.cd(1)
fImpedance = TF2("fImpedance", "60/sqrt([0])*log(1.9*(2*x+[1])/(0.8*y+[1]))", 0.04, 0.2, 0.04, 0.2)
fImpedance.SetTitle("Impedance vs trace width and distance to ground")
fImpedance.SetParameters(epsilonR, t)
fImpedance.Draw("colz")
fImpedance.GetXaxis().SetTitle("Distance to ground [mm]")
fImpedance.GetYaxis().SetTitle("Trace width [mm]")
cImpedance.cd(2)
fImpedance1D = TF1("fImpedance1D", "60/sqrt([0])*log(1.9*(2*x+[1])/(0.8*[2]+[1]))", 0.04, 0.2)
fImpedance1D.SetTitle("Impedance vs distance to ground")
fImpedance1D.SetParameters(epsilonR, t, w)
fImpedance1D.Draw()
fImpedance1D.GetXaxis().SetTitle("Distance to ground [mm]")
fImpedance1D.GetYaxis().SetTitle("Impedance [#Omega]")

# prepare the TH1
hCapTrace = []
hCapShield = []
hCapDetector = []
line_color_number = 1
line_style_number = 1
for i in range(0, len(readoutLayerRadialLengths)):
    if line_color_number == 8:
        line_color_number = 22
    if line_style_number == 8:
        line_style_number = 1
    # traces
    hCapTrace.append(TH1F())
    hCapTrace[i].SetBins(numTheta, minTheta, maxTheta)
    hCapTrace[i].SetLineColor(line_color_number)
    hCapTrace[i].SetLineStyle(line_style_number)
    hCapTrace[i].SetLineWidth(2)
    hCapTrace[i].SetTitle("Stripline capacitance; #theta; Capacitance [pF]")
    hCapTrace[i].SetName("hCapacitance_traces"+str(i))
    # shields
    hCapShield.append(TH1F())
    hCapShield[i].SetBins(numTheta, minTheta, maxTheta)
    hCapShield[i].SetLineColor(line_color_number)
    hCapShield[i].SetLineStyle(line_style_number)
    hCapShield[i].SetLineWidth(2)
    hCapShield[i].SetTitle("Signal pads - ground shields capacitance; #theta; Capacitance [pF]")
    hCapShield[i].SetName("hCapacitance_shields"+str(i))
    # area
    hCapDetector.append(TH1F())
    hCapDetector[i].SetBins(numTheta, minTheta, maxTheta)
    hCapDetector[i].SetLineColor(line_color_number)
    hCapDetector[i].SetLineStyle(line_style_number)
    hCapDetector[i].SetLineWidth(2)
    hCapDetector[i].SetTitle("Signal pad - absorber capacitance; #theta; Capacitance [pF]")
    hCapDetector[i].SetName("hCapacitance_detector"+str(i))
    if line_color_number > 8:
        line_color_number += 10
    else:
        line_color_number += 1
    line_style_number += 1

cTrace = TCanvas("cTrace", "", 600, 400)
cShield = TCanvas("cShield", "", 600, 400)
cDetector = TCanvas("cDetector", "", 600, 400)

legend = TLegend(0.1, 0.693, 0.8, 0.9)
legend.SetHeader("Longitudinal layers")
legend.SetNColumns(4)
capa_shield_max = 0
capa_det_max = 0
cellcapas = []
for i in range(0, len(readoutLayerParallelLengths)):
    print("--------------")
    for index in range(0, numTheta):
        theta = minTheta + index * deltaTheta
        thetaNext = minTheta + (index + 1) * deltaTheta
        eta = -log(tan(theta / 2.0))
        deltaEta = abs(-log(tan((minTheta + (index + 1) * deltaTheta) / 2.0)) - eta)
        if (debug):
            print("theta = ", theta)
            print("eta = ", eta)
            print("delta eta = ", deltaEta)

        # take into account the inclination in theta
        traceLength = trace_length[i] / sin(theta)
        # print("Layer %d trace length %f"%(i+1, traceLength))
        # Trace capacitance (stripline) - not used since already accounted for elsewhere
        logStripline = log(3.1 * hs / (0.8 * w + t))
        # analytical formula
        #capacitanceTrace = nmultTrace * 1 / inch2mm * 1.41 * epsilonR / logStripline * traceLength

        #JP calculate with value from measurement instead; analytical formula has assumptions that are not fulfilled
        capacitanceTrace = nmultTrace*stripLineCapaDensity*traceLength
        
        hCapTrace[i].SetBinContent(index + 1, capacitanceTrace)

        # Shield capacitance (microstrip)
        cellLength = readoutLayerParallelLengths[i] / sin(theta)
        logMicrostrip = log(5.98 * hm / (0.8 * ws + t))
        # analytical formula (nmultShield = 2)
        # capacitanceShield = nmultShield * nMergedModules[i] * cellLength * tracesPerLayer[i] * 1 / inch2mm * 0.67 * (epsilonR + 1.41) / logMicrostrip
        # from maxwell (nmultShield = 1)
        # dont multiply by nMergedThetaCells:  the shield/pad capa is reasonably independent of the cell size and the fact that there is some merging
        # done for theta cells is already taken into account by the tracesPerLayer
        capacitanceShield = nmultShield * nMergedModules[i] * cellLength * tracesPerLayer[i] * capa_per_mm[i]
        if capacitanceShield > capa_shield_max:
            capa_shield_max = capacitanceShield
        hCapShield[i].SetBinContent(index + 1, capacitanceShield)

        # Detector area (C = epsilon*A/d)
        # area = ( radius[i] * ( 1 / (tan(2. * atan(exp(- (index + 1) * deltaEta)))) -  1 / (tan(2. * atan(exp(- index * deltaEta))) ) )
        #         + radius[i + 1] * ( 1 / (tan(2. * atan(exp(- (index + 1) * deltaEta)))) -  1 / (tan(2. * atan(exp(- index * deltaEta))) ) )
        #         ) / 2. * (radius[i+1] - radius[i])
        # distance = (radius[i+1] + radius[i]) / 2. * pi / Nplanes * cos (angle) - pcbThickness / 2. - passiveThickness / 2.

        # Detector area (C = epsilon*A/d)
        area = abs(real_radial_separation[i] * (1 / tan(thetaNext) - 1 / tan(theta)) + real_radial_separation[i + 1] * (1 / tan(thetaNext) - 1 / tan(theta))
                 ) / 2. * (real_radial_separation[i+1] - real_radial_separation[i])
                 
        # get the cell size perpendicular to the plate direction from the cell size on the circle at given radius and the inclination w.r.t. radial dir, then remove the PCB and lead thickness (no need for any factor here because we are perpendicular to the PCB and lead plates) --> gives the LAr gap size perpendicular
        distance = (2 * pi * (real_radial_separation[i+1] + real_radial_separation[i]) / 2. / Nplanes * cos (inclinations_wrt_radial_dir_at_middleRadialDepth[i]) - pcbThickness - passiveThickness) / 2. # divided by two because two lar gap per cell
        distance += hhv  # the capa is between signal plate and absorber --> need to add distance between HV plate and signal pad
        distance += t  # the capa is between signal plate and absorber --> need to add distance between HV plate and signal pad
        if (abs(theta - pi / 2.) < 1e-4):
            print("LAr gap size (perpendicular) + hhv + t: %f mm" % distance)
        #capacitanceDetector = nMergedModules[i] * nMergedThetaCells[i] * 2 * epsilon0 * epsilonRLAr * area / distance  # factor 2 is because there are 2 LAr gaps for each cell
        #JP Updated to include the effect of 100um dielectric layer
        capacitanceDetector = ( nMergedModules[i] * nMergedThetaCells[i] * 2 * epsilon0 * epsilonRLAr * epsilonR * area ) / ( (distance-hhv)*epsilonR + hhv*epsilonRLAr )
        # (this is the equation for the two-dielectric sandwitch capacitors, equivalent to two capacitors in series)
        
        hCapDetector[i].SetBinContent(index + 1, capacitanceDetector)
        if capacitanceDetector > capa_det_max:
            capa_det_max = capacitanceDetector
        if (abs(theta - pi / 2.) < 1e-4):
            print("layer %d" % (i + 1), "theta=%f" % theta, ": capacitanceTrace: %.0f pF," % capacitanceTrace, "capacitanceShield: %.0f pF," % capacitanceShield, "capacitanceDetector: %.0f pF," %capacitanceDetector, "total/2: %.0f pF" % ((capacitanceTrace + capacitanceShield + capacitanceDetector)/2.))
            # , "distance %.1f mm" %distance
            cellcapas.append(round((capacitanceTrace + capacitanceShield + capacitanceDetector)/2.))

    # Draw
    cTrace.cd()
    if i == 0:
        hCapTrace[i].Draw()
    else:
        hCapTrace[i].Draw("same")
    legend.AddEntry(hCapTrace[i], "layer %d" % (i + 1), "l")
    cShield.cd()
    if i == 0:
        hCapShield[i].Draw()
    else:
        hCapShield[i].Draw("same")
    cDetector.cd()
    if i == 0:
        hCapDetector[i].Draw()
    else:
        hCapDetector[i].Draw("same")

cellcapas.reverse()

print("Cell capas per electrode:")
print(cellcapas)        

maximum = capa_shield_max

plots = TFile(filename, "RECREATE")

for i in range(0, len(readoutLayerParallelLengths)):
    hCapTrace[i].SetMinimum(0)
    hCapTrace[i].SetMaximum(maximum * 1.8)
    hCapTrace[i].Write()
    hCapShield[i].SetMinimum(0)
    hCapShield[i].SetMaximum(capa_shield_max * 1.5)
    hCapShield[i].Write()
    hCapDetector[i].SetMinimum(0)
    hCapDetector[i].SetMaximum(capa_det_max * 1.5)
    hCapDetector[i].Write()

cTrace.cd()
legend.Draw()
cTrace.Update()
cTrace.Write()
cTrace.Print("capa_trace%s.png" % apdx)
cTrace.Print("capa_trace%s.pdf" % apdx)
cShield.cd()
legend.Draw()
cShield.Update()
cShield.Write()
cShield.Print("capa_shield%s.png" % apdx)
cShield.Print("capa_shield%s.pdf" % apdx)
cDetector.cd()
legend.Draw()
cDetector.Update()
cDetector.Write()
cDetector.Print("capa_detector%s.png" % apdx)
cDetector.Print("capa_detector%s.pdf" % apdx)

fImpedance.Write()
fImpedance1D.Write()

#closeInput = raw_input("Press ENTER to exit")

