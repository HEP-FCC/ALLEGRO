#
# script to compare noise or neighbour maps in different root files
# Usage: compareMaps.py <noise/neighbours> [old_file.root] [new_file.root]")
#

import ROOT
import sys
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Compare noise or neighbour maps between two files")
parser.add_argument("maptype", type=str, help="Either noise or neighbour")
parser.add_argument("file1", type=str, help="The first file to compare")
parser.add_argument("file2", type=str, help="The second file to compare")
parser.add_argument("--nevts", type=int, default=-1, help="The number of events to process (optional; default: all)")
# parser.add_argument("--debug", action="store_true", help="If set, will print the values of the different branches")
parser.add_argument("--debugevts", type=int, default=-1, help="If >0, will print the values of the different branches for the first given number of different events")
args = parser.parse_args()
print("")

ROOT.gROOT.SetBatch()

nevts = args.nevts
file1 = args.file1
file2 = args.file2
maptype = args.maptype
# debug = args.debug
debug = args.debugevts>0
debugprint = args.debugevts

old_file = ROOT.TFile(file1)
new_file = ROOT.TFile(file2)

if maptype=="neighbours":
    treeName = "neighbours"
    branchList = ["cellId", "neighbours"]
elif maptype=="noise":
    treeName = "noisyCells"
    branchList = ["cellId", "noiseLevel", "noiseOffset"]
else:
    print("Wrong argument")
    sys.exit(1)

old_tree = old_file.Get(treeName)
new_tree = new_file.Get(treeName)

total_entries = old_tree.GetEntries()
if total_entries != new_tree.GetEntries():
    print("Trees do not have equal numbers of entries")
    print("Respectively: %lu and %lu" % (total_entries, new_tree.GetEntries()))
    sys.exit(1)
else:
    print("Trees have equal numbers of entries:", total_entries)

# decide on how many entries to run based on command line args
if nevts>0:
    total_entries = nevts

# initialise counters
badEntries = []
oldValues = {}
newValues = {}
diffs = {}
for branch in branchList:
    oldValues[branch] = []
    newValues[branch] = []
    diffs[branch] = 0

# loop over events
for i in tqdm(range(total_entries),mininterval=0.2):
    old_tree.GetEntry(i)
    new_tree.GetEntry(i)
    diff = False
    for branch in branchList:
        if branch == "neighbours":
            oldList = sorted(list(getattr(old_tree, branch)))
            newList = sorted(list(getattr(new_tree, branch)))
            if oldList != newList:
                diff = True
                diffs[branch]+=1
        else:
            if not ( getattr(old_tree, branch) == getattr(new_tree, branch) ):
                diff = True
                diffs[branch]+=1
    if diff:
        badEntries.append(i)
        if debug:
            for branch in branchList:
                old = getattr(old_tree, branch)
                new = getattr(new_tree, branch)
                if branch == "neighbours":
                    oldValues[branch].append(oldList)
                    newValues[branch].append(newList)
                else:
                    oldValues[branch].append(getattr(old_tree, branch))
                    newValues[branch].append(getattr(new_tree, branch))

print("\nNumber of different entries: ",len(badEntries))
for branch in branchList:
    print(f"Branch {branch} has {diffs[branch]} differences")

if debug:
    if len(badEntries)>0:
        nentries = min(debugprint, len(badEntries))
        print(f"\nContent of first {nentries} different entries")
        for i in range(nentries):
            print("\nEntry:", badEntries[i])
            print("Old:")
            for branch in branchList:
                print(branch, oldValues[branch][i])
            print("New:")
            for branch in branchList:
                print(branch, newValues[branch][i])

print("")
