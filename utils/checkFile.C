void  checkFile(const char* filename, int nevts=-1) {
 if (gSystem->AccessPathName(filename)){
    cout << "FILE_DOES_NOT_EXIST" << endl;
    return;
  }
  bool is_zombie = TFile(filename).IsZombie();
  if (is_zombie) {
    cout << "FILE_IS_ZOMBIE" << endl;
    return;
  }
  TFile* f = TFile::Open(filename);
  TTree* T = (TTree*) f->Get("events");
  if (T==nullptr) {
    cout << "TREE_DOES_NOT_EXIST" << endl;
    return;
  }
  if (nevts>0) {
    int entries = T->GetEntries();
    cout << "ENTRIES:" << entries << endl;
    if (entries<nevts) {
      cout << "NOT_ENOUGH_EVENTS" << endl;
    }
    else if (entries>nevts) {
      cout << "TOO_MANY_EVENTS" << endl;
    }
     else {
      cout << "OK" << endl;
    }
  }
  else {
    cout << "UNKNOWN_REFERENCE_EVENTS" << endl;
  }
}
