// script to compare xtalk maps to check if they are identical, using hashes
// much faster than python script in utils/

#include <TFile.h>
#include <TTree.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "argparse.hpp"

struct EntryData {
  Long64_t entry;
  uint64_t hash1 = 0;
  uint64_t hash2 = 0;
  Double_t noiseLevel = 0;
  Double_t noiseOffset = 0;
};
using Index = std::unordered_map<ULong64_t, EntryData>;

std::string mapType = "";
std::string refFile = "";
std::string treeName = "";
int maxDiffs = 0;
bool verbose = false;
int frequency = 500000;
int maxDebug = 0;
bool filesAreDifferent = false;

ULong64_t id1, id2;
Double_t noiseLevel1, noiseLevel2;
Double_t noiseOffset1, noiseOffset2;
std::vector<double>* values1 = nullptr;
std::vector<double>* values2 = nullptr;
std::vector<unsigned long>* neighbours1 = nullptr;
std::vector<unsigned long>* neighbours2 = nullptr;

// get tree from file
TTree* getTreeFromFile(const std::string& filename) {
  TFile* file = new TFile(filename.c_str(), "READ");
  if (file->IsZombie())
    throw std::runtime_error("Cannot open file " + filename);

  TTree* tree = nullptr;
  file->GetObject(treeName.c_str(), tree);
  if (!tree)
    throw std::runtime_error("Tree " + treeName + " not found in file " + filename);
  return tree;
}

// setup branch addresses of xtalk tree
void setupBranchesForXTalkTree(TTree* tree, ULong64_t* id, std::vector<unsigned long>*& neighbours,
                               std::vector<double>*& values) {
  tree->SetBranchAddress("cellId", id);
  tree->SetBranchAddress("list_crosstalk_neighbours", &neighbours);
  tree->SetBranchAddress("list_crosstalks", &values);
}

// setup branch addresses of noise tree
void setupBranchesForNoiseTree(TTree* tree, ULong64_t* id, Double_t* noiseLevel, Double_t* noiseOffset) {
  tree->SetBranchAddress("cellId", id);
  tree->SetBranchAddress("noiseLevel", noiseLevel);
  tree->SetBranchAddress("noiseOffset", noiseOffset);
}

// setup branch addresses of neighbours tree
void setupBranchesForNeighboursTree(TTree* tree, ULong64_t* id, std::vector<unsigned long>*& neighbours) {
  tree->SetBranchAddress("cellId", id);
  tree->SetBranchAddress("neighbours", &neighbours);
}

// setup branch addresses of given tree for file 1 or 2
void setupBranches(TTree* tree, int iFile) {
  if (mapType == "noise") {
    if (iFile == 1)
      setupBranchesForNoiseTree(tree, &id1, &noiseLevel1, &noiseOffset1);
    else
      setupBranchesForNoiseTree(tree, &id2, &noiseLevel2, &noiseOffset2);
  } else if (mapType == "xtalk") {
    if (iFile == 1)
      setupBranchesForXTalkTree(tree, &id1, neighbours1, values1);
    else
      setupBranchesForXTalkTree(tree, &id2, neighbours2, values2);
  } else if (mapType == "neighbours") {
    if (iFile == 1)
      setupBranchesForNeighboursTree(tree, &id1, neighbours1);
    else
      setupBranchesForNeighboursTree(tree, &id2, neighbours2);
  }
}

// print to output stream with optional prefix
void printWithPrefix(const std::string& prefix, std::function<void(std::ostream&)> write_actions,
                     std::ostream& out = std::cout) {
  std::stringstream ss;
  write_actions(ss);

  std::string line;
  while (std::getline(ss, line)) {
    out << prefix << line << "\n"; // ?? Prints to whichever stream you passed
  }
}

void printNoiseContent(std::ostream& os, ULong64_t id, Double_t level, Double_t offset) {
  os << "id: " << id << "\n";
  os << "noise level, offset: ";
  os << level << " " << offset << "\n";
}

void printXTalkContent(std::ostream& os, ULong64_t id, std::vector<unsigned long>*& neighbours,
                       std::vector<double>*& values) {
  os << "id: " << id << "\n";
  os << "neighbours:";
  for (auto n : *neighbours)
    os << " " << n;
  os << "\n";
  os << "xtalk values:";
  for (auto v : *values)
    os << " " << v;
  os << "\n";
}

void printNeighboursContent(std::ostream& os, ULong64_t id, std::vector<unsigned long>*& neighbours) {
  os << "id: " << id << "\n";
  os << "neighbours:";
  for (auto n : *neighbours)
    os << " " << n;
  os << "\n";
}

void printContent(std::ostream& os, int iFile) {
  if (mapType == "xtalk") {
    if (iFile == 1)
      printXTalkContent(os, id1, neighbours1, values1);
    else
      printXTalkContent(os, id2, neighbours2, values2);
  } else if (mapType == "noise") {
    if (iFile == 1)
      printNoiseContent(os, id1, noiseLevel1, noiseOffset1);
    else
      printNoiseContent(os, id2, noiseLevel2, noiseOffset2);
  } else if (mapType == "neighbours") {
    if (iFile == 1)
      printNeighboursContent(os, id1, neighbours1);
    else
      printNeighboursContent(os, id2, neighbours2);
  }
}

void printRef(std::ostream& os, Long64_t entry) {
  TFile file(refFile.c_str(), "READ");
  TTree* tree = (TTree*)file.Get(treeName.c_str());
  setupBranches(tree, 1);
  tree->GetEntry(entry);
  printContent(os, 1);
}

bool reportDifference(Long64_t entry, const std::string& message) {
  std::cerr << message << "\n";
  printWithPrefix(
      "",
      [&](std::ostream& os) {
        os << "\n--- Tree 1 (entry " << entry << ") ---\n";
        printContent(os, 1);
        os << "\n--- Tree 2 (entry " << entry << ") ---\n";
        printContent(os, 2);
        os << "\n";
      },
      std::cerr);

  filesAreDifferent = true;
  return false;
}

// fast hash value calculation (64 bit, FNV-1a algorithm)
template <typename T>
uint64_t hashVector(const std::vector<T>& v) {
  const uint64_t offset = 14695981039346656037ULL;
  const uint64_t prime = 1099511628211ULL;

  uint64_t h = offset;

  // Include size
  uint64_t size = v.size();

  const unsigned char* p = reinterpret_cast<const unsigned char*>(&size);

  for (size_t i = 0; i < sizeof(size); ++i) {
    h ^= p[i];
    h *= prime;
  }

  // Include data
  const unsigned char* data = reinterpret_cast<const unsigned char*>(v.data());

  size_t nbytes = v.size() * sizeof(T);
  for (size_t i = 0; i < nbytes; ++i) {
    h ^= data[i];
    h *= prime;
  }

  return h;
}

// Build reference map from first file
Index buildIndex(const std::string& filename) {
  std::cout << "Reading reference file " << filename << "\n";

  TTree* tree = getTreeFromFile(filename);
  tree->SetCacheSize(100 * 1024 * 1024);
  tree->AddBranchToCache("*");
  setupBranches(tree, 1);

  Long64_t n = tree->GetEntries();

  Index map;
  map.reserve(static_cast<size_t>(n * 1.3));

  auto start = std::chrono::high_resolution_clock::now();

  for (Long64_t i = 0; i < n; i++) {
    tree->GetEntry(i);
    EntryData d;
    d.entry = i;
    if (mapType == "xtalk") {
      d.hash1 = hashVector<double>(*values1);
      d.hash2 = hashVector<unsigned long>(*neighbours1);
      // map[id1] = std::make_tuple(i, hashVector<double>(*values1), hashVector<unsigned long>(*neighbours1));
    } else if (mapType == "neighbours") {
      // map[id1] = std::make_tuple(i, hashVector<unsigned long>(*neighbours1), 0);
      d.hash1 = hashVector<double>(*values1);
    } else if (mapType == "noise") {
      d.noiseLevel = noiseLevel1;
      d.noiseOffset = noiseOffset1;
    }
    map.emplace(id1, d);

    if (i % frequency == 0)
      std::cout << "  " << i << "/" << n << "\n";
  }

  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dt = stop - start;
  std::cout << "Index built in " << dt.count() << " s\n";

  return map;
}

// Compare second file to reference file
bool compareFile(const std::string& filename, const Index& reference) {
  int ndiff(0);
  int ndebug(0);

  std::cout << "\nChecking " << filename << "\n";

  TTree* tree = getTreeFromFile(filename);
  tree->SetCacheSize(100 * 1024 * 1024);
  tree->AddBranchToCache("*");

  setupBranches(tree, 2);

  Long64_t n = tree->GetEntries();
  if (n != (Long64_t)reference.size()) {
    std::cerr << "Different number of events: << " << n << " vs " << reference.size() << "\n";
    return false;
  }

  auto start = std::chrono::high_resolution_clock::now();

  // loop over entries in this file
  for (Long64_t i = 0; i < n; i++) {
    // load entry
    tree->GetEntry(i);

    // search for cell with same id in reference map
    auto it = reference.find(id2);
    if (it == reference.end()) {
      std::cerr << "Missing id " << id2 << "\n";
      return false;
    }

    // do the comparison
    bool ok(true);
    const auto& ref = it->second;
    Long64_t iref = ref.entry;
    if (mapType == "xtalk") {
      if (hashVector<double>(*values2) != ref.hash1 || hashVector<unsigned long>(*neighbours2) != ref.hash2)
        ok = false;
    } else if (mapType == "neighbours") {
      if (hashVector<unsigned long>(*neighbours2) != ref.hash1)
        ok = false;
    } else if (mapType == "noise") {
      if (noiseOffset2 != ref.noiseOffset || noiseLevel2 != ref.noiseLevel)
        ok = false;
    }
    if (!ok) {
      ndiff++;
      ndebug++;
      if (ndebug <= maxDebug) {
        printWithPrefix(
            "[DEBUG] ",
            [&](std::ostream& os) {
              os << "Mismatch for id " << id2 << " corresponding to entry " << iref << " in file1"
                 << " and to entry " << i << " in file2\n\n";
              os << "\n--- Tree 1 (entry " << iref << ") ---\n";
              printRef(os, iref);
              os << "\n--- Tree 2 (entry " << i << ") ---\n";
              printContent(os, 2);
              os << "\n";
            },
            std::cerr);
        if (ndebug == maxDebug) {
          std::cerr << "[DEBUG] Maximum number of debug messages reached, will suppress further output\n\n";
        }
      }
      if (ndiff == maxDiffs)
        break;
    }

    // print counter
    if (i % frequency == 0) {
      ndebug++;
      std::cout << "  " << i << "/" << n << "\n";
      if (verbose && ndebug <= maxDebug) {
        printWithPrefix(
            "[DEBUG] ",
            [&](std::ostream& os) {
              os << "\n--- Tree 1 (entry " << iref << ") ---\n";
              printRef(os, iref);
              os << "\n--- Tree 2 (entry " << i << ") ---\n";
              printContent(os, 2);
              os << "\n";
            },
            std::cout);
        if (ndebug == maxDebug) {
          std::cerr << "[DEBUG] Maximum number of debug messages reached, will suppress further output\n\n";
        }
      }
    }
  }

  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dt = stop - start;
  std::cout << "Comparison finished in " << dt.count() << " s\n";
  std::cout << "Number of differences: " << ndiff << "\n";
  return (ndiff == 0);
}

// compare files assuming same sorting of events
// return as soon as there is a difference
bool compareSortedFiles(const std::string& filename1, const std::string& filename2) {
  std::cout << "\nPerforming entry-by-entry comparison assuming same sorting\n";
  int ndebug(0);

  // preparing trees
  TTree* tree1 = getTreeFromFile(filename1);
  TTree* tree2 = getTreeFromFile(filename2);
  setupBranches(tree1, 1);
  setupBranches(tree2, 2);

  // check that number of entries is the same
  Long64_t n = tree1->GetEntries();
  if (tree2->GetEntries() != n) {
    std::cerr << "Different number of entries\n";
    exit(1);
  }

  // compare entries one by one
  auto start = std::chrono::high_resolution_clock::now();
  for (Long64_t i = 0; i < n; ++i) {
    tree1->GetEntry(i);
    tree2->GetEntry(i);
    if (id1 != id2) {
      std::cerr << "Different cellId at entry " << i << " (" << id1 << " vs " << id2 << "),"
                << " files might be sorted differently\n";
      return false;
    }

    if (mapType == "xtalk") {
      if (values1->size() != values2->size() || neighbours1->size() != neighbours2->size()) {
        return reportDifference(i, "Different vector size for same cellId");
      }
      if (std::memcmp(values1->data(), values2->data(), values1->size() * sizeof(double)) != 0) {
        return reportDifference(i, "Different neighbour/xtalk vector contents for same cellId");
      }
      if (std::memcmp(neighbours1->data(), neighbours2->data(), neighbours1->size() * sizeof(unsigned long)) != 0) {
        return reportDifference(i, "Different neighbour/xtalk vector contents for same cellId");
      }
    }

    else if (mapType == "noise") {
      if ((noiseLevel1 != noiseLevel2) or (noiseOffset1 != noiseOffset2)) {
        return reportDifference(i, "Different noise values for same cellId");
      }
    }

    else if (mapType == "neighbours") {
      if (neighbours1->size() != neighbours2->size()) {
        return reportDifference(i, "Different neighbours vector size for same cellId");
      }
      if (std::memcmp(neighbours1->data(), neighbours2->data(), neighbours1->size() * sizeof(unsigned long)) != 0) {
        return reportDifference(i, "Different neighbour vector contents for same cellId");
      }
    }

    if (i % frequency == 0) {
      ndebug++;
      std::cout << "  " << i << "/" << n << "\n";
      if (verbose && ndebug <= maxDebug) {
        printWithPrefix(
            "[DEBUG] ",
            [&](std::ostream& os) {
              os << "\n--- Tree 1 (entry " << i << ") ---\n";
              printContent(os, 1);
              os << "\n--- Tree 2 (entry " << i << ") ---\n";
              printContent(os, 2);
              os << "\n";
            },
            std::cout);
        if (ndebug == maxDebug) {
          std::cerr << "[DEBUG] Maximum number of debug messages reached, will suppress further output\n\n";
        }
      }
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dt = stop - start;
  std::cout << "Comparison finished in " << dt.count() << " s\n";

  return true;
}

// main
int main(int argc, char** argv) {
  // create argument parser
  argparse::ArgumentParser parser("compareTrees", "1.0");
  parser.add_description("Compare noise, neighbour or cross-talk neighbour maps between two files");
  parser.add_argument("maptype").help("Either noise, neighbour or xtalk").choices("noise", "neighbours", "xtalk");
  parser.add_argument("file1").help("The first file to compare");
  parser.add_argument("file2").help("The second file to compare");
  parser.add_argument("-m", "--max-diffs")
      .help("The maximum number of diffs before the program stops")
      .scan<'i', int>()
      .default_value(maxDiffs);
  parser.add_argument("-v", "--verbose").help("Print verbose output").default_value(false).implicit_value(true);
  parser.add_argument("-i", "--ignoreSorting")
      .help("Ignore the sorting of the trees, and only match entries by key (comparison will be done using hashes)")
      .default_value(false)
      .implicit_value(true);
  parser.add_argument("-d", "--debug-events")
      .help("If >0, will print the values of the different branches for the first given number of different events")
      .scan<'i', int>()
      .default_value(maxDebug);
  parser.add_argument("-f", "--frequency")
      .help("The frequency of the output messages")
      .scan<'i', int>()
      .default_value(frequency);

  // parse the arguments
  try {
    parser.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << std::endl;
    std::cerr << parser; // This automatically prints the help menu on error!
    return 1;
  }

  // retrieve values safely
  mapType = parser.get<std::string>("maptype");
  refFile = parser.get<std::string>("file1");
  std::string file2 = parser.get<std::string>("file2");
  maxDiffs = parser.get<int>("--max-diffs");
  verbose = parser.get<bool>("--verbose");
  maxDebug = parser.get<int>("--debug-events");
  frequency = parser.get<int>("--frequency");
  bool ignoreSorting = parser.get<bool>("--ignoreSorting");

  // determine name of tree based on type of map
  if (mapType == "neighbours") {
    treeName = "neighbours";
  } else if (mapType == "noise") {
    treeName = "noisyCells";
  } else if (mapType == "xtalk") {
    treeName = "crosstalk_neighbours";
  }

  if (verbose) {
    std::cout << "[DEBUG] Verbose mode is on.\n";
  }

  std::cout << "File1: " << refFile << std::endl;
  std::cout << "File2: " << file2 << std::endl;

  // do the comparison
  try {
    // first, try sequential comparison
    bool ok(false);

    if (not ignoreSorting) {
      ok = compareSortedFiles(refFile, file2);
      if (ok) {
        std::cout << "\nTrees are identical\n";
        return 0;
      }
      if (filesAreDifferent) {
        std::cout << "\nTrees are different\n";
        return 2;
      }
      std::cout << "\nDifferent CellIDs found for given entry, Trees are different or sorted in a different way\n";
    }

    // if it fails (different ordering, try with hash/map-based comparison
    std::cout << "\nTrying index-based comparison\n";
    auto reference = buildIndex(refFile);
    ok = compareFile(file2.c_str(), reference);
    if (ok) {
      std::cout << "\nTrees are identical\n";
      return 0;
    } else {
      std::cout << "\nTrees are different\n";
      return 2;
    }
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
