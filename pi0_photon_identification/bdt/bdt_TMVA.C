#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TFile.h"
#include "TTree.h"

void bdt_TMVA() {
  auto outputFile = TFile::Open("TMVA_ClassificationOutput.root", "RECREATE");

  TMVA::Factory factory("TMVAClassification", outputFile,
			"!V:ROC:!Silent:Color:!DrawProgressBar:AnalysisType=Classification" );

  TString inputFileNameSig = "/home/lit/public/giovanni/sample/strip_L3_1M/production_reconstruction_particle_gamma_1M.root";
  TString inputFileNameBkg = "/home/lit/public/giovanni/sample/strip_L3_1M/production_reconstruction_particle_pi0_1M.root";

  auto inputFileSig = TFile::Open( inputFileNameSig );
  TTree *signalTree     = (TTree*)inputFileSig->Get("events");

  auto inputFileBkg = TFile::Open( inputFileNameBkg );
  TTree *backgroundTree = (TTree*)inputFileBkg->Get("events");

  TMVA::DataLoader * loader = new TMVA::DataLoader("dataset");

  // global event weights per tree (see below for setting event-wise weights)
  Double_t signalWeight     = 1.0;
  Double_t backgroundWeight = 1.0;

  // You can add an arbitrary number of signal or background trees
  loader->AddSignalTree    ( signalTree,     signalWeight     );
  loader->AddBackgroundTree( backgroundTree, backgroundWeight );

  loader->AddVariable( "AugmentedCaloClusters.energy[0]", "Ecl", "GeV", 'F' );
  for (int i=0; i<189; i++)
    loader->AddVariable( Form("_AugmentedCaloClusters_shapeParameters[%d]",i), Form("var%d",i), "", 'F' );

  // Apply additional cuts on the signal and background samples (can be different)
  TCut mycut = "AugmentedCaloClusters.energy[0] > 25 && AugmentedCaloClusters.energy[0]<50";
  // TCut mycut = "";

  loader->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );


  //Boosted Decision Trees
  factory.BookMethod(loader,TMVA::Types::kBDT, "BDT",
		     "nTrees=1000:MaxDepth=4:BoostType=AdaBoost:AdaBoostBeta=1:SeparationType=GiniIndex:nCuts=20:MinNodeSize=1:PruneMethod=NoPruning");

  factory.TrainAllMethods();
  // Evaluate the BDT
  factory.TestAllMethods();
  factory.EvaluateAllMethods();
  // Save the output
  //factory.Close();
  // Close your TFile objects
  inputFileBkg->Close();
  inputFileSig->Close();
}
