//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Oct 16 21:44:10 2025 by ROOT version 6.36.04
// from TChain events/
//////////////////////////////////////////////////////////

#ifndef autoCalib_analytic_topo_h
#define autoCalib_analytic_topo_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "vector"
#include "podio/ObjectID.h"
#include "podio/detail/LinkFwd.h"
#include "edm4hep/ClusterData.h"

class autoCalib_analytic_topo {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.
   static constexpr Int_t kMaxECalEndcapTurbineContributions = 700005;
   static constexpr Int_t kMax_ECalEndcapTurbineContributions_particle = 700005;
   static constexpr Int_t kMaxECalEndcapTurbinePositioned = 15470;
   static constexpr Int_t kMaxECalEndcapTurbinePositionedSimCaloHitLinks = 15470;
   static constexpr Int_t kMax_ECalEndcapTurbinePositionedSimCaloHitLinks_from = 15470;
   static constexpr Int_t kMax_ECalEndcapTurbinePositionedSimCaloHitLinks_to = 15470;
   static constexpr Int_t kMaxEMECCaloClusters = 1;
   static constexpr Int_t kMax_EMECCaloClusters_clusters = 1;
   static constexpr Int_t kMax_EMECCaloClusters_hits = 1;
   static constexpr Int_t kMaxEMECCaloTopoClusterCells = 154700;
   static constexpr Int_t kMaxEMECCaloTopoClusters = 24300;
   static constexpr Int_t kMax_EMECCaloTopoClusters_clusters = 10;
   static constexpr Int_t kMax_EMECCaloTopoClusters_hits = 154700;
   static constexpr Int_t kMaxEventHeader = 1;
   static constexpr Int_t kMaxMCParticles = 80;
   static constexpr Int_t kMax_MCParticles_parents = 70;
   static constexpr Int_t kMax_MCParticles_daughters = 70;

   // Declaration of leaf types
   Int_t           EMECCaloTopoClusterCells_;
   ULong_t         EMECCaloTopoClusterCells_cellID[kMaxEMECCaloTopoClusterCells];   //[EMECCaloTopoClusterCells_]
   Float_t         EMECCaloTopoClusterCells_energy[kMaxEMECCaloTopoClusterCells];   //[EMECCaloTopoClusterCells_]
   Int_t           EMECCaloTopoClusters_;
   UInt_t          EMECCaloTopoClusters_hits_begin[kMaxEMECCaloTopoClusters];   //[EMECCaloTopoClusters_]
   UInt_t          EMECCaloTopoClusters_hits_end[kMaxEMECCaloTopoClusters];   //[EMECCaloTopoClusters_]
  Float_t         EMECCaloTopoClusters_energy[kMaxEMECCaloTopoClusters];   //[EMECCaloTopoClusters_]
   Float_t         EMECCaloTopoClusters_position_z[kMaxEMECCaloTopoClusters];  //[EMECCaloTopoClusters_]

   vector<string>  *GPIntKeys;
   vector<vector<int> > *GPIntValues;
   vector<string>  *GPFloatKeys;
   vector<vector<float> > *GPFloatValues;
   vector<string>  *GPDoubleKeys;
   vector<vector<double> > *GPDoubleValues;
   vector<string>  *GPStringKeys;
   vector<vector<string> > *GPStringValues;

   // List of branches
   TBranch        *b_EMECCaloTopoClusterCells_;   //!
   TBranch        *b_EMECCaloTopoClusterCells_cellID;   //!
   TBranch        *b_EMECCaloTopoClusterCells_energy;   //!
   TBranch        *b_EMECCaloTopoClusters_;   //!
   TBranch        *b_EMECCaloTopoClusters_hits_begin;   //!
   TBranch        *b_EMECCaloTopoClusters_hits_end;   //!
   TBranch        *b_EMECCaloTopoClusters_position_z;   //!
   TBranch        *b_EMECCaloTopoClusters_energy;   //!
  
   TBranch        *b_GPIntKeys;   //!
   TBranch        *b_GPIntValues;   //!
   TBranch        *b_GPFloatKeys;   //!
   TBranch        *b_GPFloatValues;   //!
   TBranch        *b_GPDoubleKeys;   //!
   TBranch        *b_GPDoubleValues;   //!
   TBranch        *b_GPStringKeys;   //!
   TBranch        *b_GPStringValues;   //!

   autoCalib_analytic_topo(TTree *tree=0);
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Calibrate();
   virtual bool     Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef autoCalib_analytic_topo_cxx
autoCalib_analytic_topo::autoCalib_analytic_topo(TTree *tree) : fChain(0) 
{

  Init(tree);
}



Int_t autoCalib_analytic_topo::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t autoCalib_analytic_topo::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void autoCalib_analytic_topo::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   GPIntKeys = 0;
   GPIntValues = 0;
   GPFloatKeys = 0;
   GPFloatValues = 0;
   GPDoubleKeys = 0;
   GPDoubleValues = 0;
   GPStringKeys = 0;
   GPStringValues = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);
   
   fChain->SetBranchStatus("*", 0);
   fChain->SetBranchStatus("EMECCaloTopoClusterCells", 1);
   fChain->SetBranchAddress("EMECCaloTopoClusterCells", &EMECCaloTopoClusterCells_, &b_EMECCaloTopoClusterCells_);
   fChain->SetBranchStatus("EMECCaloTopoClusterCells.cellID", 1);
   fChain->SetBranchAddress("EMECCaloTopoClusterCells.cellID", EMECCaloTopoClusterCells_cellID, &b_EMECCaloTopoClusterCells_cellID);
   fChain->SetBranchStatus("EMECCaloTopoClusterCells.energy", 1);
   fChain->SetBranchAddress("EMECCaloTopoClusterCells.energy", EMECCaloTopoClusterCells_energy, &b_EMECCaloTopoClusterCells_energy);
   fChain->SetBranchStatus("EMECCaloTopoClusters", 1);
   fChain->SetBranchAddress("EMECCaloTopoClusters", &EMECCaloTopoClusters_, &b_EMECCaloTopoClusters_);
   fChain->SetBranchStatus("EMECCaloTopoClusters.hits_begin", 1);
   fChain->SetBranchAddress("EMECCaloTopoClusters.hits_begin", EMECCaloTopoClusters_hits_begin, &b_EMECCaloTopoClusters_hits_begin);
   fChain->SetBranchStatus("EMECCaloTopoClusters.hits_end", 1);
   fChain->SetBranchAddress("EMECCaloTopoClusters.hits_end", EMECCaloTopoClusters_hits_end, &b_EMECCaloTopoClusters_hits_end);
   fChain->SetBranchStatus("EMECCaloTopoClusters.energy", 1);
   fChain->SetBranchAddress("EMECCaloTopoClusters.energy", EMECCaloTopoClusters_energy, &b_EMECCaloTopoClusters_energy);
   fChain->SetBranchStatus("EMECCaloTopoClusters.position.z", 1);
   fChain->SetBranchAddress("EMECCaloTopoClusters.position.z", EMECCaloTopoClusters_position_z, &b_EMECCaloTopoClusters_position_z);
   fChain->SetBranchStatus("GPIntKeys", 1);
   fChain->SetBranchAddress("GPIntKeys", &GPIntKeys, &b_GPIntKeys);
   fChain->SetBranchStatus("GPIntValues", 1);
   fChain->SetBranchAddress("GPIntValues", &GPIntValues, &b_GPIntValues);
   fChain->SetBranchStatus("GPFloatKeys", 1);
   fChain->SetBranchAddress("GPFloatKeys", &GPFloatKeys, &b_GPFloatKeys);
   fChain->SetBranchStatus("GPFloatValues", 1);
   fChain->SetBranchAddress("GPFloatValues", &GPFloatValues, &b_GPFloatValues);
   fChain->SetBranchStatus("GPDoubleKeys", 1);
   fChain->SetBranchAddress("GPDoubleKeys", &GPDoubleKeys, &b_GPDoubleKeys);
   fChain->SetBranchStatus("GPDoubleValues", 1);
   fChain->SetBranchAddress("GPDoubleValues", &GPDoubleValues, &b_GPDoubleValues);
   fChain->SetBranchStatus("GPStringKeys", 1);
   fChain->SetBranchAddress("GPStringKeys", &GPStringKeys, &b_GPStringKeys);
   fChain->SetBranchStatus("GPStringValues", 1);
   fChain->SetBranchAddress("GPStringValues", &GPStringValues, &b_GPStringValues);
  
   Notify();
}

bool autoCalib_analytic_topo::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return true;
}

void autoCalib_analytic_topo::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t autoCalib_analytic_topo::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef autoCalib_analytic_topo_cxx
