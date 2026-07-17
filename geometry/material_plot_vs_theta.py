from __future__ import print_function
import argparse

from plotstyle import FCCStyle

from math import tanh, acos, degrees
from array import array

import ROOT
ROOT.gROOT.SetBatch(ROOT.kTRUE)

import os

def main():
    parser = argparse.ArgumentParser(
        description='Material plotter vs theta or cos(theta)')
    parser.add_argument('--fname', "-f", dest='fname',
                        default="out_material_scan.root", type=str, help="name of file to read")
    # parser.add_argument('--cosThetaMax', "-c", dest='cosThetaMax', default=1.0, type=float, help="maximum cosine of polar angle")
    # parser.add_argument('--thetaMin', "-t", dest='thetaMin', default=0.0, type=float, help="minimum polar angle in degrees")
    parser.add_argument('--suffix', "-s", dest='suffix',
                        default="", type=str, help="suffix of output")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cosThetaMax', "-c", dest='cosThetaMax',
                       default=argparse.SUPPRESS, type=float, help="maximum cosine of polar angle")
    group.add_argument('--thetaMin', "-t", dest='thetaMin', default=argparse.SUPPRESS,
                       type=float, help="minimum polar angle in degrees")

    args = parser.parse_args()
    suffix = args.suffix

    thetaMin = -1.
    cosThetaMax = -1.
    if 'cosThetaMax' in args or 'm' in args:
        cosThetaMax = float(args.cosThetaMax)
    else:
        thetaMin = float(args.thetaMin)

    f = ROOT.TFile.Open(args.fname, "read")
    tree = f.Get("materials")
    histDict = {}

    bins = []
    # retrieve the eta bins of the material scan
    # and convert to theta or cos(theta)
    for etaBin, entry in enumerate(tree):
        cosTheta = tanh(entry.eta)
        theta = degrees(acos(cosTheta))
        if abs(cosTheta) < 1e-9:
            if cosThetaMax > -1.:
                bins.append(0.0)
            else:
                bins.append(90.0)
        else:
            if cosThetaMax > -1.:
                bins.append(cosTheta)
            else:
                bins.append(theta)
    if thetaMin > -1.:
        bins.reverse()
    print(bins)

    # go through the eta bins and fill the histograms in the histDict, skipping air
    # keys in the histDict are the material names
    for etaBin, entry in enumerate(tree):
        nMat = entry.nMaterials
        for i in range(nMat):
            if entry.material.at(i) == "Air":
                continue
            if entry.material.at(i) not in histDict.keys():
                histDict[entry.material.at(i)] = {
                    "x0": ROOT.TH1F("", "", len(bins) - 1, array("f", bins)),
                    "lambda": ROOT.TH1F("", "", len(bins) - 1, array("f", bins)),
                    "depth": ROOT.TH1F("", "", len(bins) - 1, array("f", bins)),
                }
            hs = histDict[entry.material.at(i)]
            if cosThetaMax > -1:
                hs["x0"].SetBinContent(
                    etaBin + 1, hs["x0"].GetBinContent(etaBin + 1) + entry.nX0.at(i))
                hs["lambda"].SetBinContent(
                    etaBin + 1, hs["lambda"].GetBinContent(etaBin + 1) + entry.nLambda.at(i))
                hs["depth"].SetBinContent(
                    etaBin + 1, hs["depth"].GetBinContent(etaBin + 1) + entry.matDepth.at(i))
            else:
                binToFill = len(bins) - etaBin - 1
                hs["x0"].SetBinContent(
                    binToFill, hs["x0"].GetBinContent(binToFill) + entry.nX0.at(i))
                hs["lambda"].SetBinContent(
                    binToFill, hs["lambda"].GetBinContent(binToFill) + entry.nLambda.at(i))
                hs["depth"].SetBinContent(binToFill, hs["depth"].GetBinContent(
                    binToFill) + entry.matDepth.at(i))

    axis_titles = ["Number of X_{0}",
                   "Number of #lambda", "Material depth [cm]"]

    # This loop does the drawing, sets the style and saves the pdf files
    os.makedirs("plots/material/", exist_ok=True)
    #if not os.path.isdir("plots"):
    #    os.mkdir("plots")

    for plot, title in zip(["x0", "lambda", "depth"], axis_titles):
        legend = ROOT.TLegend(.25, .65, .44, .94)
        legend.SetLineColor(0)
        ths = ROOT.THStack()
        for i, material in enumerate(histDict.keys()):
            linecolor = 1
            if i >= len(FCCStyle.fillcolors):
                i = i % len(FCCStyle.fillcolors)

            fillcolor = FCCStyle.fillcolors[i]
            histDict[material][plot].SetLineColor(linecolor)
            histDict[material][plot].SetFillColor(fillcolor)
            histDict[material][plot].SetLineWidth(1)
            histDict[material][plot].SetFillStyle(1001)

            ths.Add(histDict[material][plot])
            legend.AddEntry(histDict[material][plot], material, "f")

        # debug
        # hcont = ROOT.TH1F(ths.GetStack().Last())
        # for j in range(hcont.GetNbinsX()):
        #    print(hcont.GetBinLowEdge(j + 1),
        #          hcont.GetBinLowEdge(j + 1) + hcont.GetBinWidth(j + 1),
        #          hcont.GetBinContent(j + 1))
        cv = ROOT.TCanvas()
        if cosThetaMax > -1:
            haxis = ROOT.TH2F("haxis", "haxis", 1, -args.cosThetaMax,
                              args.cosThetaMax, 1, 0.0, 1.5 * ths.GetMaximum())
            haxis.GetXaxis().SetTitle("cos(#theta)")
        else:
            haxis = ROOT.TH2F("haxis", "haxis", 1, args.thetaMin,
                              180. - args.thetaMin, 1, 0.0, 1.5 * ths.GetMaximum())
            haxis.GetXaxis().SetTitle("#theta [#circ]")
        haxis.Draw("AXIS")
        haxis.GetYaxis().SetTitle(title)
        ths.Draw("SAME")
        haxis.Draw("AXISSAME")
        legend.Draw()
        if cosThetaMax > -1:
            cv.Print("plots/material/" + plot + "_vs_costheta" + suffix + ".pdf")
            cv.Print("plots/material/" + plot + "_vs_costheta" + suffix + ".png")
        else:
            cv.Print("plots/material/" + plot + "_vs_theta" + suffix + ".pdf")
            cv.Print("plots/material/" + plot + "_vs_theta" + suffix + ".png")

        if cosThetaMax > -1:
            haxis2 = ROOT.TH2F("haxis2", "haxis2", 1, 0,
                               args.cosThetaMax, 1, 0.0, 1.5 * ths.GetMaximum())
            haxis2.GetXaxis().SetTitle("cos(#theta)")
        else:
            haxis2 = ROOT.TH2F("haxis2", "haxis2", 1, args.thetaMin,
                               90.0, 1, 0.0, 1.5 * ths.GetMaximum())
            haxis2.GetXaxis().SetTitle("#theta [#circ]")
        haxis2.Draw("AXIS")
        haxis2.GetYaxis().SetTitle(title)
        ths.Draw("SAME")
        haxis2.Draw("AXISSAME")
        legend.Draw()
        if cosThetaMax > -1:
            cv.Print("plots/material/" + plot + "_vs_costheta_pos" + suffix + ".pdf")
            cv.Print("plots/material/" + plot + "_vs_costheta_pos" + suffix + ".png")
        else:
            cv.Print("plots/material/" + plot + "_vs_theta_pos" + suffix + ".pdf")
            cv.Print("plots/material/" + plot + "_vs_theta_pos" + suffix + ".png")


if __name__ == "__main__":
    FCCStyle.initialize()
    main()
