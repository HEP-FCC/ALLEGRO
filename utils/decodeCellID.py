import ROOT
import dd4hep
import os
import sys

systemEB = 4
systemEEC = 5
systemHB = 8
systemHEC = 9

# ------------------------------------------------------------------
# Parse cellIDs from command line (space-separated)
# ------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python decodeCellID.py <cellID1> [cellID2 ...]")
    print("Example: python decodeCellID.py 0x834 12345")
    sys.exit(1)

cellIDs = [int(arg, 0) for arg in sys.argv[1:]]


# ------------------------------------------------------------------
# Load detector geometry
# ------------------------------------------------------------------
compactFile = "FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml"
path_to_detector = os.environ.get("K4GEO", "")
detectorFile = path_to_detector + "/" + compactFile
detector = dd4hep.Detector.getInstance()
detector.fromXML(detectorFile)
print("Loaded detector from compact file:", detectorFile)

# ------------------------------------------------------------------
# Loop over cellIDs
# ------------------------------------------------------------------
previous_readout = ""
coder = None
for cellID in cellIDs:
    print("\n====================================")
    print("CellID:", hex(cellID), f"({cellID})")

    # Get system
    system = cellID & 0b11111
    if system==systemEB:
        readoutName = "ECalBarrelModuleThetaMerged"
    elif system==systemEEC:
        readoutName = "ECalEndcapTurbine"
    elif system==systemHB:
        readoutName = "HCalBarrelReadout"
    elif system==systemHEC:
        readoutName = "HCalEndcapReadout"
    else:
        print("Unknown system", system)
        continue

    # Get readout (use caching)
    if readoutName != previous_readout:
        readout = detector.readout(readoutName)
        encoding = readout.idSpec().fieldDescription()
        coder = ROOT.dd4hep.BitFieldCoder(encoding)
        previous_readout = readoutName

    print("System:", system)
    print("Readout:", readoutName)
    print("CellID encoding:", encoding)

    # Decode and print all fields
    for field in coder.fields():
        name = field.name()
        value = coder.get(cellID, name)
        print(f"{name}: {value}")
