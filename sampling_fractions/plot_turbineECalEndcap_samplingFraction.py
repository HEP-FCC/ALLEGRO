import calo_init
import json
import math

#python plot_samplingFraction.py /eos/user/b/brfranco/rootfile_storage/202011_condor_calib_5kEvt/calibration_output_pdgID_22_pMin_?_pMax_?_thetaMin_90_thetaMax_90.root 1 10 50 100 -r 1000 10000 50000 100000 --totalNumLayers 12 --preview -outputfolder plots_sampling_fraction_201124 --plotSFvsEnergy
#python plot_samplingFraction.py /eos/user/b/brfranco/rootfile_storage/202011_condor_calib_5kEvt/calibration_output_pdgID_22_pMin_50000_pMax_50000_thetaMin_?_thetaMax_?.root 50 70 90 -r 50 70 90 --totalNumLayers 12 --preview -outputfolder plots_sampling_fraction_theta_50GeV --plotSFvsEnergy --theta
calo_init.add_defaults()
calo_init.parser.add_argument("--merge", help="merge layers", default = [1] * 98, nargs='+') # bin 0 is empty! (before calo)
calo_init.parser.add_argument("-t","--title", default="Sampling fraction", help="Graph title", type=str)
calo_init.parser.add_argument("-n","--histogramName", default="ecal_sf_layer", help="Name of the histogram with sampling fraction (postfixed with number of layer)", type = str)
calo_init.parser.add_argument("--histogramNameMean", default="ecal_sf", help="Name of the histogram with sampling fraction (sufixed with number of layer)", type = str)
calo_init.parser.add_argument("-max","--axisMax", help="Maximum of the axis", type = float)
calo_init.parser.add_argument("-min","--axisMin", help="Minimum of the axis", type = float)
calo_init.parser.add_argument("-outputfolder",default="plots_sampling_fraction", help="Name of the output foler for the plots", type = str)
calo_init.parser.add_argument("--totalNumLayers", default = 98, type = int)
calo_init.parser.add_argument("--numFirstLayer", default = 0, help="ID of first layer used in histograms name", type = int)
calo_init.parser.add_argument("--layerWidth", default = [3.532000] * 50 + [6.75]*50 + [12.818]*50 , type = float)
calo_init.parser.add_argument("--X0density", default = 0.422, help="Xo density of a current detector (X0/cm)", type = float)
calo_init.parser.add_argument("--roundBrackets", help="Use round brackets for unit", action = 'store_true')
calo_init.parser.add_argument("--preview", help="Plot preview of fits", action = 'store_true')
calo_init.parser.add_argument("--plotSFvsEnergy", help="Plot sf as a function of energy", action = 'store_true')
calo_init.parser.add_argument("--theta", help="Plot sf as a function of theta instead of energy, plotSFvsEnergy must also be set to true", action = 'store_true')
calo_init.parser.add_argument("--sed", help="Modify the sampling fraction in the FCCSW python configs", action = 'store_true')
calo_init.parser.add_argument("--json", help="Store the sampling fractions in a json file", type=str, default = '')
calo_init.parser.add_argument("--specialLabel", help="Additional label to be plotted", type=str, default = "")
calo_init.parse_args()
calo_init.print_config()

histName = calo_init.args.histogramName
histNameMean = calo_init.args.histogramNameMean

from ROOT import gSystem, gROOT, TCanvas, TH1F, TGraphErrors, TF1, gStyle, kRed, kBlue, kGray, TFile, TTree, TPad, TGaxis, gPad, TLine, TColor, kTRUE, nullptr, TLegend
from draw_functions import prepare_graph, prepare_divided_canvas,  prepare_single_canvas, draw_text, draw_1histogram
import numpy
from math import sqrt, ceil, floor
import os

#gStyle.SetImageScaling(3.)
#gStyle.SetOptFit(1111) # fi you want all fiut info on the poreview canvas
gROOT.SetBatch(kTRUE)

if not os.path.isdir(calo_init.args.outputfolder):
    os.mkdir(calo_init.args.outputfolder)

merge = [sum(calo_init.args.merge[:i]) for i in range(0,len(calo_init.args.merge))]
sliceWidth = calo_init.args.layerWidth  # cm
if len(sliceWidth) == 1:
    sliceWidth = sliceWidth * len(merge)
sliceSum = []
sumWidths = 0
for width in sliceWidth:
    sumWidths += width
    sliceSum.append(sumWidths)
print(('sliceWidths',sliceWidth))
startIndex = calo_init.args.numFirstLayer
Nslices = calo_init.args.totalNumLayers
if sum(calo_init.args.merge) != Nslices:
    print(('Number of total layers (',Nslices,') is not the same as a sum of "--merge" arguments (sum = ',sum(calo_init.args.merge),')'))
    exit(0)
Nslicesmerged = len(merge)
all_graphs = []
graphTitles = []
avgSF = []
avgSFerr = []

colour = ['#4169E1','#D2691E','#228B22','#DC143C','#696969','#9932CC','#D2B48C', 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
colour = [TColor.GetColor(c) for c in colour]
dict_layer_sfVSenergyGraph = {}
for islice in range(startIndex, Nslices + startIndex):
    dict_layer_sfVSenergyGraph[islice] = TGraphErrors()

# first get all the resolutions and prepare graphs
for ifile, filename in enumerate(calo_init.filenamesIn):
    energy = calo_init.energy(ifile)
    f = TFile(filename, "READ")
    # mean value of the sampling fraction
    hMean = f.Get(histNameMean)
    print("hMean name: ", hMean.GetName())
    print("hMean title: ", hMean.GetTitle())
    print("hMean nentries: ", hMean.GetEntries())
    print("hMean mean: ", hMean.GetMean())
    print("hMean x axis min: ", hMean.GetXaxis().GetBinLowEdge(0))
    print("hMean x axis max: ", hMean.GetXaxis().GetBinUpEdge(hMean.GetNbinsX()))
    if hMean:
        fitPre = TF1("fitPre","gaus", hMean.GetMean() - 1. * hMean.GetRMS(), hMean.GetMean() + 1. * hMean.GetRMS())
        resultPre = hMean.Fit(fitPre, "SQRN")
        print("prefit OK")
        print("prefit parameters: ", resultPre.Get().Parameter(1), " ", resultPre.Get().Parameter(2) )
        fit = TF1("fit","gaus",resultPre.Get().Parameter(1) - 2. * resultPre.Get().Parameter(2), resultPre.Get().Parameter(1) + 2. * resultPre.Get().Parameter(2) )
        result = hMean.Fit(fit, "SQRN")
        print("fit OK")
        avgSF.append(result.Get().Parameter(1))
        avgSFerr.append(result.Get().Parameter(2))
    hmerged = []
    # first merge adjacent layers and get histograms of SF
    for islice in range(startIndex, Nslices + startIndex):
        h = TH1F() 
        h = f.Get(histName+str(islice))
        print("h ", histName+str(islice), "  is a ", type(h))
        # if first hist to be merged
        lastIm = -1
        if islice - startIndex in merge:
            lastIm += 1
            hmerged.append(h)
        else:
            hmerged[lastIm].Add(h)
    gSF = TGraphErrors()
    gSF_eta = TGraphErrors()
    # now fit SF with Gaussians
    if calo_init.args.preview:
        cPreview = prepare_divided_canvas('preview_e'+str(energy)+'GeV', 'Preview for '+str(energy)+'GeV', Nslicesmerged)
        fitoptions = "SQR"
    else:
        fitoptions = "SQRN"
    for islice, h in enumerate(hmerged):
      #  h.Rebin(5)
        # get rid of 0's
        h.SetBinContent(1,0.)
        h.Print()
        fitPre = TF1("fitPre","gaus", h.GetMean() - 1. * h.GetRMS(), h.GetMean() + 1. * h.GetRMS())
        print("prefit parameters: ", resultPre.Get().Parameter(1), " ", resultPre.Get().Parameter(2) )
        #h.Rebin(10)
        resultPre = h.Fit(fitPre, fitoptions)
        fit = TF1("fit","gaus",resultPre.Get().Parameter(1) - 2. * resultPre.Get().Parameter(2), resultPre.Get().Parameter(1) + 2. * resultPre.Get().Parameter(2) )
        result = h.Fit(fit, fitoptions)
        factor = 1.9
        while not result:
            print ("Hi everyone!!!")
            fit = TF1("fit","gaus",resultPre.Get().Parameter(1) - factor * resultPre.Get().Parameter(2), resultPre.Get().Parameter(1) + factor * resultPre.Get().Parameter(2) )
            result = h.Fit(fit, fitoptions)
            factor = factor - 0.1
          #  print(factor, " ", result)
            if factor < 0.2:
                break
        # print(result)        
        if result and result.Ndf() > 0:
            # if it fits terribly, try to fit in narrower range
            print("chi2 is ", result.Chi2() / result.Ndf())
            if result.Chi2() / result.Ndf() > 10:
                print("Bad fit chi2 %f"%(result.Chi2() / result.Ndf()))
                refit = TF1("refit","gaus",resultPre.Get().Parameter(1) - resultPre.Get().Parameter(2), resultPre.Get().Parameter(1) + resultPre.Get().Parameter(2) )
                result = h.Fit(refit, fitoptions)
                print("New fit chi2 %f"%(result.Chi2() / result.Ndf()))
        # make graph
        if result.Get():
            gSF.SetPoint(islice, sliceSum[islice]-sliceWidth[islice] * 0.5, result.Get().Parameter(1))
            gSF.SetPointError(islice, sliceWidth[islice] * 0.5 , result.Get().ParError(1))
            sliceRho = sliceSum[islice]-sliceWidth[islice] * 0.5
            sliceTheta = math.atan2(sliceRho, 348)
            sliceEta = -math.log(math.tan(sliceTheta/2.))
            gSF_eta.SetPoint(islice, sliceEta  , result.Get().Parameter(1))
            gSF_eta.SetPointError(islice, 0. , result.Get().ParError(1))
            #if islice < len(merge) - 1:
            #   gSF.SetPoint(islice, (merge[islice] + 0.5 * (merge[islice + 1] - merge[islice])) * sliceWidth, result.Get().Parameter(1))
            #   gSF.SetPointError(islice, 0.5 * (merge[islice + 1] - merge[islice]) * sliceWidth , result.Get().Parameter(2))
            #else:
            #    gSF.SetPoint(islice, (merge[islice] + 0.5 * (merge[islice] - merge[islice - 1])) * sliceWidth, result.Get().Parameter(1))
            #    gSF.SetPointError(islice, 0.5 * (merge[islice] - merge[islice - 1]) * sliceWidth , result.Get().Parameter(2))
            dict_layer_sfVSenergyGraph[islice].SetPoint(ifile, energy, result.Get().Parameter(1))
            dict_layer_sfVSenergyGraph[islice].SetPointError(ifile, 0, result.Get().ParError(1))
        if calo_init.args.preview: # draw both singla canvas and one big divided canvas (later is buggy at the moment)
            tmp_canvas = prepare_single_canvas("energy_%s_"%(energy) + h.GetTitle().replace(" ", "_"), "energy_%s_"%(energy) + h.GetTitle().replace(" ", "_"))
            draw_1histogram(h,"","")
            tmp_canvas.SaveAs(os.path.join(calo_init.args.outputfolder, "preview_" + tmp_canvas.GetTitle() + ".png" ))
            cPreview.cd(islice+1)
            draw_1histogram(h,"","")
    if calo_init.args.preview:
        cPreview.SaveAs(os.path.join(calo_init.args.outputfolder, "preview_sampling_fraction_energy_%d.png"%energy))
    prepare_graph(gSF, 'sf_'+str(len(merge))+'layers', ';radial depth [cm];sampling fraction', ifile+9)

    all_graphs.append(gSF)
   # prepare_graph(gSF_eta, 'sf_'+str(len(merge))+'layers', ';eta;sampling fraction', ifile+9)
   # all_graphs.append(gSF_eta)
    graphTitles.append('#color['+str(colour[ifile])+']{'+str(energy)+' GeV e^{-}}')
    string_for_fccsw = ""
    for islice in range(0, Nslicesmerged):
        if islice > 0:
            string_for_fccsw += " + "
        string_for_fccsw += "["+str(gSF.GetY()[islice])+"] * "+str(calo_init.args.merge[islice])
    print("Sampling fraction for energy %d: "%energy)
    print(string_for_fccsw)

if calo_init.args.sed:
    command = "sed -i 's/samplingFraction =.*,/samplingFraction = %s,/' "%string_for_fccsw
    os.system(command + " run*AndCaloSim.py") # it has to be launched from FCCSW_ecal folder
    print(command + " run*AndCaloSim.py")
    os.system("sed -i 's/samplingFractions =.*,/samplingFractions = %s,/' fcc_ee_upstream_inclinedEcal.py"%string_for_fccsw) # it has to be launched from FCCSW_ecal folder
    os.system("sed -i 's/SFfcc =.*/SFfcc = %s/' ../geometry/create_noise_file* caloNtupleAnalyzer/energy_vs_depth_wrt_noise.py"%string_for_fccsw) # it has to be launched from FCCSW_ecal folder
    #print(command + " ../../k4RecCalorimeter/RecFCCeeCalorimeter/tests/options/* ../../FCCSW/Examples/options/run_calo_fullsim_fccee.py")

# MN: Add json output
if calo_init.args.json:
    with open(calo_init.args.json, 'w') as jsonfile:
        to_json = []
        for islice in range(0, Nslicesmerged):
            to_json.extend( [gSF.GetY()[islice]] * calo_init.args.merge[islice] )
        json.dump(to_json, jsonfile)


canv = prepare_single_canvas('sf_e'+str(energy)+'GeV', 'Sampling fraction for '+str(energy)+'GeV')

# Draw graph and all labels
prepare_graph(gSF, 'sf_'+str(len(merge))+'layers', ';radial depth [cm];sampling fraction', ifile+9)
#prepare_graph(gSF_eta, 'sf_'+str(len(merge))+'layers', ';eta;sampling fraction', ifile+9)
all_graphs[0].Draw("ape")
for g in all_graphs[1:]:
    g.Draw("pe")
if calo_init.args.axisMax:
    all_graphs[0].SetMaximum(calo_init.args.axisMax)
if calo_init.args.axisMin:
    all_graphs[0].SetMinimum(calo_init.args.axisMin)
canv.Update()

lines = []
for iLine, line in enumerate(avgSF):
    lines.append(TLine(0, avgSF[iLine], 65, avgSF[iLine]))
    lines[iLine].SetLineColor(colour[iLine])
    all_graphs[iLine].SetMarkerColor(colour[iLine])
    all_graphs[iLine].SetLineColor(colour[iLine])
    lines[iLine].Draw('same')

if len(graphTitles) > 1:
    draw_text(graphTitles, [0.18,0.9 - 0.07 * len(graphTitles),0.4,0.95], 0.4, 0).SetTextSize(0.06)

# Draw all labels
if calo_init.args.specialLabel:
    draw_text([calo_init.args.specialLabel], [0.57,0.88, 0.85,0.98], kGray+3, 0).SetTextSize(0.05)
canv.Update()

# Draw sf vs energy graph
if calo_init.args.plotSFvsEnergy:
    for islice in range(startIndex, Nslices + startIndex):
        dict_layer_sfVSenergyGraph[islice].GetXaxis().SetRangeUser(calo_init.energy(0) -10, calo_init.energy(-1) + calo_init.energy(-1)/10.0)
        if calo_init.args.theta:
            x_axis_label = "#Theta angle [degrees]"
            graph_title = "Sampling fraction versus polar angle: layer %d"%islice
        else:
            x_axis_label = "Energy [GeV]"
            graph_title = "Sampling fraction versus energy: layer %d"%islice
        dict_layer_sfVSenergyGraph[islice].GetXaxis().SetTitle(x_axis_label)
        dict_layer_sfVSenergyGraph[islice].GetYaxis().SetTitle("Sampling fraction")
        dict_layer_sfVSenergyGraph[islice].GetYaxis().SetRangeUser(0, 0.42)
        prepare_graph(dict_layer_sfVSenergyGraph[islice], 'sf_vs_energy_layer%d'%islice, graph_title)
        #canvas = TCanvas('sf_vs_energy_layer%d'%islice, 'sf_vs_energy_layer%d'%islice)
        canvas = prepare_single_canvas('sf_vs_energy_layer%d'%islice, 'sf_vs_energy_layer%d'%islice)
        canvas.SetTicky(1)
        dict_layer_sfVSenergyGraph[islice].Draw('ape')
        # fit the graph
        fit = dict_layer_sfVSenergyGraph[islice].Fit('pol1', 'SQ', "", 0, calo_init.energy(-1) + 0.1 * calo_init.energy(-1))
        b = fit.Parameter(0)
        a = fit.Parameter(1)
        legend = TLegend(0.23, 0.2, 0.52, 0.35)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.AddEntry(nullptr,  'y = ax + b', "")
        legend.AddEntry(nullptr,  'a = ' + str(round(a, 6)), "")
        legend.AddEntry(nullptr,  'b = ' + str(round(b, 4)), "")
        legend.Draw()
        layer_for_file_name = str(islice)
        if islice < 10:
            layer_for_file_name = '0'+str(islice)
        canvas.Print(os.path.join(calo_init.args.outputfolder,"sampling_fraction_vs_energy_layer%s.png"%layer_for_file_name))

# Save canvas and root file with graph, const term and sampling term
if calo_init.output(0):
    canv.SaveAs(calo_init.output(0)+".pdf")
    canv.SaveAs(calo_init.output(0)+".png")
    plots = TFile(calo_init.output(0)+".root","RECREATE")
    if calo_init.args.preview:
        cPreview.SaveAs("preview_"+calo_init.output(0)+".png")
else:
    canv.SaveAs(os.path.join(calo_init.args.outputfolder, "sampling_fraction_plots.pdf"))
    canv.SaveAs(os.path.join(calo_init.args.outputfolder, "sampling_fraction_plots.png"))
    plots = TFile(os.path.join(calo_init.args.outputfolder,"sampling_fraction.root"),"RECREATE")
for g in all_graphs:
    g.Write()

mean = numpy.zeros(1, dtype=float)
std = numpy.zeros(1, dtype=float)
t = TTree("samplingFraction", "Sampling fraction for detector layers")
t.Branch("mean", mean, "mean/D");
t.Branch("std", std, "std/D");
for islice in range(0, Nslicesmerged):
    for ilay in range(0, calo_init.args.merge[islice]):
        mean[0] = gSF.GetY()[islice]
        std[0] = gSF.GetErrorY(islice)
        t.Fill()
plots.Write()
plots.Close()

#raw_input("Press ENTER to exit")
