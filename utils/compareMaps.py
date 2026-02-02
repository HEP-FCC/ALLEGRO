#
# script to compare noise or neighbour maps in different root files
# Usage: compareMaps.py <noise/neighbours> [file1.root] [file2.root]")
#

import ROOT
import sys
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
    description="Compare noise or neighbour maps between two files"
)
parser.add_argument("maptype", type=str, help="Either noise or neighbour")
parser.add_argument("file1", type=str, help="The first file to compare")
parser.add_argument("file2", type=str, help="The second file to compare")
parser.add_argument(
    "--nevts", type=int, default=-1,
    help="The number of events to process (optional; default: all)"
)
parser.add_argument(
    "--verbose", action="store_true",
    help="Print verbose output (full file content)"
)
parser.add_argument(
    "--ignoreCounts", action="store_true",
    help="Ignore that the trees have different counts and match entries by key"
)
parser.add_argument(
    "--debugevts", type=int, default=-1,
    help="If >0, will print the values of the different branches for the first given number of different events"
)
args = parser.parse_args()
print("")

ROOT.gROOT.SetBatch()

nevts = args.nevts
filename1 = args.file1
filename2 = args.file2
maptype = args.maptype
# debug = args.debug
debug = args.debugevts>0
debugprint = args.debugevts
verbose = args.verbose
ignoreCounts = args.ignoreCounts

file1 = ROOT.TFile(filename1)
file2 = ROOT.TFile(filename2)

if maptype=="neighbours":
    treeName = "neighbours"
    branchList = ["cellId", "neighbours"]
elif maptype=="noise":
    treeName = "noisyCells"
    branchList = ["cellId", "noiseLevel", "noiseOffset"]
elif maptype=="xtalk":
    treeName = "crosstalk_neighbours"
    branchList = ["cellId", "list_crosstalk_neighbours", "list_crosstalks", "CellInfo"]
else:
    print("Wrong argument")
    sys.exit(1)

tree1 = file1.Get(treeName)
tree2 = file2.Get(treeName)

total_entries = tree1.GetEntries()
if total_entries != tree2.GetEntries():
    print("Trees do not have equal numbers of entries")
    print("Respectively: %lu and %lu" % (total_entries, tree2.GetEntries()))
    if not ignoreCounts:
        print("Exiting..")
        sys.exit(1)
    else:
        print("Ignoring..")
else:
    print("Trees have equal numbers of entries:", total_entries)


# decide on how many entries to run based on command line args
if nevts>0:
    total_entries = min(total_entries, nevts)

# standard search: same number of entries, assume sorted in same way
if not ignoreCounts:
    # initialise counters
    badEntries = []
    file1Values = {}
    file2Values = {}
    diffs = {}
    for branch in branchList:
        file1Values[branch] = []
        file2Values[branch] = []
        diffs[branch] = 0

    # loop over events
    for i in tqdm(range(total_entries),mininterval=0.2):
        tree1.GetEntry(i)
        tree2.GetEntry(i)
        diff = False
        if verbose: print("Entry", i)
        for branch in branchList:
            if branch == "neighbours":
                file1List = sorted(list(getattr(tree1, branch)))
                file2List = sorted(list(getattr(tree2, branch)))
                if file1List != file2List:
                    diff = True
                    diffs[branch]+=1
                if verbose:
                    print("Branch", branch)
                    print("  file1: ", file1List)
                    print("  file2: ", file2List)
            else:
                if not ( getattr(tree1, branch) == getattr(tree2, branch) ):
                    diff = True
                    diffs[branch]+=1
                if verbose:
                    print("Branch", branch)
                    print("  file1: ", getattr(tree1, branch))
                    print("  file2: ", getattr(tree2, branch))

        if diff:
            badEntries.append(i)
            if debug:
                for branch in branchList:
                    if branch == "neighbours":
                        file1Values[branch].append(file1List)
                        file2Values[branch].append(file2List)
                    else:
                        file1Values[branch].append(getattr(tree1, branch))
                        file2Values[branch].append(getattr(tree2, branch))

    print("\nNumber of different entries: ",len(badEntries))
    for branch in branchList:
        print(f"Branch {branch} has {diffs[branch]} differences")

    if debug:
        if len(badEntries)>0:
            nentries = min(debugprint, len(badEntries))
            print(f"\nContent of first {nentries} different entries")
            for i in range(nentries):
                print("\nEntry:", badEntries[i])
                print("File 1:")
                for branch in branchList:
                    print(branch, file1Values[branch][i])
                print("File 2:")
                for branch in branchList:
                    print(branch, file2Values[branch][i])

else:
    # index second tree by the key (cellId) for fast lookup
    index2 = {}
    for j in range(tree2.GetEntries()):
        tree2.GetEntry(j)
        key = getattr(tree2, "cellId")
        index2[key] = j
    # same for first tree
    index1 = {}
    for i in range(tree1.GetEntries()):
        tree1.GetEntry(i)
        key = getattr(tree1, "cellId")
        index1[key] = i

    badEntries = []
    missingEntries1 = []
    missingEntries2 = []
    goodEntries = 0
    diffs = {b: 0 for b in branchList}

    # check entries in second tree but not in first one
    for j in tqdm(range(tree2.GetEntries()), mininterval=0.2):
        tree2.GetEntry(j)
        key2 = getattr(tree2, "cellId")
        if key2 not in index1:
            # no matching event in new tree
            missingEntries1.append((j,key2))
            continue
        
    # check entries in first tree but not in second one
    for i in tqdm(range(tree1.GetEntries()), mininterval=0.2):
        tree1.GetEntry(i)
        key1 = getattr(tree1, "cellId")
        if key1 not in index2:
            # no matching event in new tree
            missingEntries2.append((i,key1))
            continue

    print("\nNumber of entries present in 1st file but missing in 2nd one:", len(missingEntries2))
    if len(missingEntries2)>0:
        print("List:")
        for (entry, cellid) in missingEntries2:
            print(f"Entry = {entry}, cellID = {cellid}")
    print("\nNumber of entries present in 2nd file but missing in 1st one:", len(missingEntries1))
    if len(missingEntries1)>0:
        print("List:")
        for (entry, cellid) in missingEntries1:
            print(f"Entry = {entry}, cellID = {cellid}")

    # now compare common entries (cellID in both files) - slow, so might want to run over only a fraction of events
    # iterate first tree, match by key
    for i in tqdm(range(total_entries), mininterval=0.2):
        tree1.GetEntry(i)
        key1 = getattr(tree1, "cellId")
        if key1 not in index2:
            # no matching event in new tree
            continue

        # beware: sloooow
        j = index2[key1]
        tree2.GetEntry(j)

        diff = False
        for branch in branchList:
            if branch == "neighbours":
                file1List = sorted(list(getattr(tree1, branch)))
                file2List = sorted(list(getattr(tree2, branch)))
                if file1List != file2List:
                    diff = True
                    diffs[branch] += 1
            else:
                if getattr(tree1, branch) != getattr(tree2, branch):
                    diff = True
                    diffs[branch] += 1

        if diff:
            badEntries.append((i, j))
            # stop early if not debugging
            if debugprint <= 0:
                break
        else:
            goodEntries+=1

    print(f"\nRan over {total_entries} entries")
    print("\nNumber of entries common to both files and with same content", goodEntries)
    print("\nNumber of entries common to both files but with differences:", len(badEntries))
    for branch in branchList:
        print(f"Branch {branch} has {diffs[branch]} differences")

    if debugprint > 0 and badEntries:
        nentries = min(debugprint, len(badEntries))
        print(f"\nDetails of first {nentries} mismatches:")
        for idx in range(nentries):
            i, j = badEntries[idx]
            print("\nFile 1:", i, "File 2:", j)
            tree1.GetEntry(i)
            tree2.GetEntry(j)
            for branch in branchList:
                val1 = getattr(tree1, branch)
                val2 = getattr(tree2, branch)
                print(f"{branch}: old={tree1}  new={tree2}")

print("")
