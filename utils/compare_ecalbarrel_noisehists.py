import ROOT
import sys
import math
if len(sys.argv) != 3:
    print("Usage: python compare.py file1.root file2.root")
    sys.exit(1)
f1 = ROOT.TFile.Open(sys.argv[1])
f2 = ROOT.TFile.Open(sys.argv[2])
n_layers = 11
hist_names = [
    f"h_elecNoise_fcc_{d}" for d in range (1, n_layers)
]

tolerance = 1e-9  # adjust if needed
returnCode = 0
for name in hist_names:
    h1 = f1.Get(name)
    h2 = f2.Get(name)
    if not h1 or not h2:
        print(f"[ERROR] Missing histogram: {name}")
        returnCode = 1
        continue
    if (h1.GetNbinsX() != h2.GetNbinsX() or
        h1.GetNbinsY() != h2.GetNbinsY()):
        print(f"[ERROR] Different binning in {name}")
        returnCode = 1
        continue
    max_diff = 0.0
    n_diff = 0
    for ix in range(1, h1.GetNbinsX() + 1):
        for iy in range(1, h1.GetNbinsY() + 1):
            v1 = h1.GetBinContent(ix, iy)
            v2 = h2.GetBinContent(ix, iy)
            diff = abs(v1 - v2)
            if diff > tolerance:
                n_diff += 1
                if n_diff < 10:  # limit printout
                    print(f"{name}: diff at ({ix},{iy}) -> {v1} vs {v2}")
            if diff > max_diff:
                max_diff = diff                
    if n_diff == 0:
        print(f"[OK] {name}: identical within tolerance")
    else:
        print(f"[DIFF] {name}: {n_diff} bins differ, max diff = {max_diff}")
        returnCode = 1
f1.Close()
f2.Close()
sys.exit(returnCode)
