#ifndef autoCalib_analytic_topo_cxx
#define autoCalib_analytic_topo_cxx
#include "autoCalib_analytic_topo.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TVectorD.h>
#include <TDecompLU.h>
#include <iostream>
#include <fstream>
#include <Math/Minimizer.h>
#include <Math/Functor.h>
#include <Minuit2/Minuit2Minimizer.h>

int getSystem(uint64_t cellID ) {
  return cellID & 0xf;
}

int getWheel(uint64_t cellID) {
  
  cellID = cellID >> 13;
  return (cellID & 0x7);

}

int getLayer(uint64_t cellID) {
  
  cellID = cellID >> 16;
  return (cellID & 0xfff);

}

#include <TMatrixD.h>

#define nLayers 98
#define Etrue 40.0



void autoCalib_analytic_topo::Calibrate()
{

  
  float etotperlayer[nLayers] = {};
  TH1F *h_e_mode[nLayers];
  TH1F *h_e_mean[nLayers];
  
  TH2F* khist;
  TH1F* chist;
  
  double k[nLayers*nLayers] = {};
  double c[nLayers] = {};
  double v[nLayers] = {};
  
  std::vector<double> eperlayer[nLayers];
  std::vector<double> barrelE;

  if (fChain == 0) return;
  
  Long64_t nentries = fChain->GetEntriesFast();
  
  Long64_t nbytes = 0, nb = 0;

  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
  
    for (int iCluster = 0; iCluster < EMECCaloTopoClusters_; iCluster++) {
      double e[nLayers], eb = 0.;
      for (int i = 0; i < nLayers; i++) {
        e[i] = 0;
      }
      if (TMath::Abs(EMECCaloTopoClusters_position_z[iCluster]) < 3200) continue;
      if (EMECCaloTopoClusters_energy[iCluster] < 1.) continue;
       for (unsigned iCell = EMECCaloTopoClusters_hits_begin[iCluster]; iCell < EMECCaloTopoClusters_hits_end[iCluster]; iCell++) {
        if (getSystem(EMECCaloTopoClusterCells_cellID[iCell]) == 5) {
          int layer = getLayer(EMECCaloTopoClusterCells_cellID[iCell]) & 0xfff;
          if (layer  > nLayers) {
            std::cout << "cellID is " << EMECCaloTopoClusterCells_cellID[iCell] << std::endl;
            std::cout << "Layer = " << layer << std::endl;
            continue;
          }
          int wheel = getWheel(EMECCaloTopoClusterCells_cellID[iCell]);
          if (TMath::IsNaN(EMECCaloTopoClusterCells_energy[iCell])) {
            std::cout << "got nan for layer " << layer << std::endl;
            std::cout << "Cell ID is " << hex << EMECCaloTopoClusterCells_cellID[iCell] << dec << std::endl;
          }
          e[layer] += EMECCaloTopoClusterCells_energy[iCell];
          etotperlayer[layer] += EMECCaloTopoClusterCells_energy[iCell];
	} else if (getSystem(EMECCaloTopoClusterCells_cellID[iCell]) == 4) {
          eb += EMECCaloTopoClusterCells_energy[iCell];
	}
       }
      for (int i = 0; i < nLayers; i++) {
        eperlayer[i].push_back(e[i]);
      }
      barrelE.push_back(eb);
    }
    
  }

  for (int l = 0; l < nLayers; l++) {
    if (etotperlayer[l] == 0.) {
      std::cout << "No energy in layer " << l << std::endl;
    }
  }

  // now calculate all the coefficients and other values needed
  khist = new TH2F("k", "k", nLayers, 0, nLayers, nLayers, 0, nLayers);
  chist = new TH1F("c", "c", nLayers, 0, nLayers);
  
  for (unsigned l = 0; l < nLayers; l++ ) {
    v[l] = 0;
    for (unsigned i = 0; i < eperlayer[0].size(); i++ ) {
      v[l] += (Etrue-barrelE[i])*eperlayer[l][i];
      for (int j = 0; j < nLayers; j++) {
        k[l*nLayers+j] += eperlayer[l][i]*eperlayer[j][i];      
      }
    }
  }

  for (int l = 0; l < nLayers; l++ ) {
    if (etotperlayer[l] == 0.) {
      k[l*nLayers + l] = 1.;
    }
    // check for any other 0 diagonal elements
    if (abs(k[l*nLayers + l]) < 1e-12) {
      k[l*nLayers + l] = 1.;
      std::cout << "Fixing up 0 diagonal element" << std::endl;
    }
    for (int j = 0; j < nLayers; j++) {      
      khist->SetBinContent(l, j, k[l*nLayers+j]);
    }
  }

  
  TMatrixD kmat(nLayers, nLayers, k);
  TMatrixD vmat( nLayers, 1, v);
  
  TMatrixD cmat( nLayers, 1, c);
  
  for (int i  =0 ; i < nLayers; i++) {
    if (abs(kmat(i,i))< 1.e-9) {
      std::cout << "Diagonal is 0 for " << i << std::endl;
    }
  }

  Double_t elements[nLayers] = {0};
  TVectorD b(nLayers, elements);
  
  Bool_t goodSol;
  
  cmat = (kmat.Invert())*vmat;
  
  std::cout << "ecalEndcapSamplingFraction = ";
  for (int i  =0 ; i < nLayers-1; i++) {
    if (cmat(i,0) != 0.) {
      std::cout << "[" << 1./cmat(i,0) << "] * 1 + ";
    } else {
      std::cout << "[1.0] * 1 + ";
    }
  }

  std::ofstream f_out("samplingFractionsAnalytic.txt");
  f_out << "ecalEndcapSamplingFraction = ";
  for (int i  =0 ; i < nLayers; i++) {
    float sf;
    if (cmat(i,0) == 0) {
      sf = 1.;
      chist->SetBinContent(i, 0);
    } else {
      sf = 1./cmat(i,0);
    }
    chist->SetBinContent(i, sf);
    f_out << "[" << sf << "] * 1";
    if (i == nLayers-1) {
      f_out << "" << std::endl;
    } else {
      f_out << "+ ";
    }
  }
}

#endif
