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

#include "getSystem.cpp"
#include "getLayer.cpp"
#include "getWheel.cpp"

#include <TMatrixD.h>

////#define nLayers 1160
#define nLayers 98
#define Etrue 40.0

TH1F *h_e_mode[nLayers];
TH1F *h_e_mean[nLayers];

TH2F* khist;
TH1F* chist;

double k[nLayers*nLayers] = {};
double c[nLayers] = {};
double v[nLayers] = {};

//autoCalib_analytic_topo* a;

std::vector<double> eperlayer[nLayers];
std::vector<double> barrelE;


void autoCalib_analytic_topo::Calibrate()
{
//   In a ROOT session, you can do:
//      root> .L autoCalib_analytic.C
//      root> autoCalib_analytic t
//      root> t.GetEntry(12); // Fill t data members with entry number 12
//      root> t.Show();       // Show values of entry 12
//      root> t.Show(16);     // Read and show values of entry 16
//      root> t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch

  float etotperlayer[nLayers] = {};
  
  if (fChain == 0) return;
  
  Long64_t nentries = fChain->GetEntriesFast();
  
  Long64_t nbytes = 0, nb = 0;

  //  std::cout << "Just getting started" << std::endl;
  
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;
    for (int iCluster = 0; iCluster < EMECCaloTopoClusters_; iCluster++) {
      //  std::cout << "iCluster = " << iCluster << std::endl;
      double e[nLayers], eb = 0.;
      for (int i = 0; i < nLayers; i++) {
        e[i] = 0;
      }
      if (TMath::Abs(EMECCaloTopoClusters_position_z[iCluster]) < 3200) continue;
      if (EMECCaloTopoClusters_energy[iCluster] < 1.) continue;
       for (int iCell = EMECCaloTopoClusters_hits_begin[iCluster]; iCell < EMECCaloTopoClusters_hits_end[iCluster]; iCell++) {
	 //	 std::cout << "iCell = " << iCell << std::endl;
        if (getSystem(EMECCaloTopoClusterCells_cellID[iCell]) == 5) {
          int layer = getLayer(EMECCaloTopoClusterCells_cellID[iCell]) & 0xfff;
	  //std::cout << "Layer = " << layer << std::endl;
          if (layer  > nLayers) {
            std::cout << "cellID is " << EMECCaloTopoClusterCells_cellID[iCell] << std::endl;
            std::cout << "Layer = " << layer << std::endl;
          //  //  sleep(1);
            continue;
          }
          int wheel = getWheel(EMECCaloTopoClusterCells_cellID[iCell]);
          if (TMath::IsNaN(EMECCaloTopoClusterCells_energy[iCell])) {
            std::cout << "got nan for layer " << layer << std::endl;
            std::cout << "Cell ID is " << hex << EMECCaloTopoClusterCells_cellID[iCell] << dec << std::endl;
            std::cout << "Call position is " << EMECCaloTopoClusterCells_position_x[iCell] <<  " " << EMECCaloTopoClusterCells_position_y[iCell] <<  " " <<  EMECCaloTopoClusterCells_position_z[iCell] <<  " " << std::endl;
            sleep(5);
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

  std::cout << "Got here" << std::endl;
  
  for (int l = 0; l < nLayers; l++) {
    if (etotperlayer[l] == 0.) {
      std::cout << "No energy in layer " << l << std::endl;
    }
  }

  // now calculate all the coefficients and other values needed
  khist = new TH2F("k", "k", nLayers, 0, nLayers, nLayers, 0, nLayers);
  chist = new TH1F("c", "c", nLayers, 0, nLayers);
  
  for (int l = 0; l < nLayers; l++ ) {
    v[l] = 0;
    for (int i = 0; i < eperlayer[0].size(); i++ ) {
      v[l] += (Etrue-barrelE[i])*eperlayer[l][i];
      for (int j = 0; j < nLayers; j++) {
        if (i==0) {
          //      k[l*nLayers+j] = 0.;
        }
        k[l*nLayers+j] += eperlayer[l][i]*eperlayer[j][i];      
      }
    }
  }

  std::cout << "Got here too" << std::endl;
   
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
      //std::cout << l << " " << j << " " << k[l*nLayers+j] << std::endl;
      khist->SetBinContent(l, j, k[l*nLayers+j]);
      // if (k[l*nLayers+j] < 1.0E-9) {
        //      std::cout << "Resetting 0 for " << l << ", " << j << " " << k[l*nLayers+j] << std::endl;
        //k[l*nLayers+j] = 0.2;
      // }
    }
  }

  std::cout << "Got here as well" << std::endl;
  
  TMatrixD kmat(nLayers, nLayers, k);
  TMatrixD vmat( nLayers, 1, v);
  
  TMatrixD cmat( nLayers, 1, c);
  
  std::cout << "Just checking... vmat(50,0) = " << vmat(50, 0) << std::endl;
  std::cout << "Just checking... kmat(50,45) = " << kmat(50, 45) << std::endl;
  std::cout << "Just checking... kmat(45,50) = " << kmat(45, 50) << std::endl;
  for (int i  =0 ; i < nLayers; i++) {
    if (abs(kmat(i,i))< 1.e-9) {
      std::cout << "Diagonal is 0 for " << i << std::endl;
    }
  }

  Double_t elements[nLayers] = {0};
  TVectorD b(nLayers, elements);
  
  //  TDecompLU lu(kmat);
  Bool_t goodSol;
  std::cout << "About to solve" << std::endl;
  //lu.Solve(b, goodSol);
  
  
  cmat = (kmat.Invert())*vmat;
  //cmat = kmat*vmat;
  std::cout << "Solved!" << std::endl;
  
  std::cout << "ecalEndcapSamplingFraction = ";
  for (int i  =0 ; i < nLayers-1; i++) {
    if (cmat(i,0) != 0.) {
      std::cout << "[" << 1./cmat(i,0) << "] * 1 + ";
    } else {
      std::cout << "[1.0] * 1 + ";
    }
  }
  
  std::cout << "["<< 1./cmat(nLayers-1,0) << "] * 1" << std::endl;
  
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
    // if (sf < 0.0 || sf > 1.0) sf = 1.0;
    chist->SetBinContent(i, sf);
    f_out << "[" << sf << "] * 1";
    if (i == nLayers-1) {
      f_out << "" << std::endl;
    } else {
      f_out << "+ ";
    }
  }
}

