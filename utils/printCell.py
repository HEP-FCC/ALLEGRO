#
# printCellNeighboursAndNoise.py
#
# print content of noise and neighbour maps
# TODO: consolidate, merge with decodeCellID.py
#
import ROOT
import dd4hep
import os
import sys
import argparse

# ================================
# CONFIG
# ================================
# systemIDs
systemEB = 4
systemEEC = 5
systemHB = 8
systemHEC = 9

# systems that we want to decode
systems = [systemEB, systemEEC, systemHB, systemHEC]

# names of the readout for each system
readoutMap = {
    4 : "ECalBarrelModuleThetaMerged",
    5 : "ECalEndcapTurbine",
    8 : "HCalBarrelReadout",
    9 : "HCalEndcapReadout",
}

# default encoding (can be overridden reading from the XML)
defaultEncodingMap = {
    4 : "system:0:4,cryo:4:1,type:5:3,subtype:8:3,layer:11:8,module:19:11,theta:30:10",
    5 : "system:0:4,cryo:4:1,type:5:3,subtype:8:3,side:11:-2,wheel:13:3,layer:16:12,module:28:11,rho:39:8,z:47:8",
    8 : "system:0:4,layer:4:5,row:9:9,theta:18:9,phi:27:10",
    9 : "system:0:4,type:4:3,layer:7:6,row:13:11,theta:24:11,phi:35:10"
}

# default files
defaultFilenameNeighbours = "neighbours_map_HCalBarrel.root"
defaultFilenameNoise = "cellNoise_map_electronicsNoiseLevel_ecalB_thetamodulemerged_hcalB_thetaphi.root"
treenameNeighbours = "neighbours"
treenameNoise = "noisyCells"

# geometry file (if enconding is to be read from the xml file)
defaultCompactFile = "FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml"


# ====================================
# Return readout name for given system
# ====================================
def readoutStr(system):
    if system in readoutMap:
        return readoutMap[system]
    else:
        print("Unknown system", system)
        readoutName = ""
    return readoutName


coderMap = {}


# ================================
# Load decoders
# ================================
def readCodersFromXML(compactFile):
    path_to_detector = os.environ.get("K4GEO", "")
    detectorFile = path_to_detector + "/" + compactFile
    det = dd4hep.Detector.getInstance()
    det.fromXML(detectorFile)
    for system in systems:
        readout = det.readout(readoutStr(system))
        coderMap[system] = readout.idSpec().decoder()

def loadCoders(readFromFile=False, compactFile=""):
    if readFromFile:
        print("Reading encoding from compact files")
        readCodersFromXML(compactFile)
    else:
        print("Using default encodings")
        for system in systems:
            encoding = defaultEncodingMap[system]
            coderMap[system] = ROOT.dd4hep.BitFieldCoder(encoding)

    print("\nLoaded encoding maps:")
    print("{:8s} {:30s} {:s}".format("System","readout","encoding"))
    for system in systems:
        print("{:<8d} {:30s} {:s}".format(system, readoutStr(system), coderMap[system].fieldDescription()))
    print("\n" + "="*60)


# ================================
# Decode cell information
# ================================
previous_system = 0
coder = None
def decode(cellID):
    global previous_system, coder
    """Return dict with all fields"""

    # Get system
    # 5 bits
    # system = cellID & 0b11111
    # 4 bits
    system = cellID & 0b1111
    readoutName = readoutStr(system)
    if readoutName == "": return

    # Get coder (use caching)
    if system != previous_system:
        coder = coderMap[system]
        previous_system = system

    # Decode and print all fields
    # for field in coder.fields():
    #     name = field.name()
    #     value = coder.get(cellID, name)
    #     print(f"{name}: {value}")
    
    fields = {}
    for field in coder.fields():
        name = field.name()
        fields[name] = coder.get(cellID, name)
    return fields


# =============================================
# Print information for cell with given cell ID
# =============================================
def print_cell(cellID):
    print("cellID:", int(cellID))

    fields = decode(cellID)

    for k, v in fields.items():
        print(f"{k}: {v}")

    print()


# =======================================
# Print cell with position iEntry in tree
# =======================================
def print_entry(iEntry, showNeighbours=True, showNoise=False):
    TNeighbours.GetEntry(iEntry)

    # print("="*50)
    print()
    cID = TNeighbours.cellId
    print_cell(cID)

    if showNeighbours:
        print("Neighbours:\n")
        # neighbours = TNeighbours.neighbours
        # neihbours = sorted(list(getattr(TNeighbours, "neighbours")))
        for n in neighbours:
            print_cell(n)

    if showNoise and TNoise:
        for j in range(TNoise.GetEntries()):
            TNoise.GetEntry(j)
            cIDNoise = TNoise.cellId
            if cIDNoise == cID:
                noiseLevel = TNoise.noiseLevel
                noiseOffset = TNoise.noiseOffset
                print(f"Noise: level={noiseLevel}, offset={noiseOffset}")
                break

    print("="*50)


# =========================================
# Loop over neighbours and print their info
# =========================================
def print_neighbours_of_cell(cellID):
    for i in range(TNeighbours.GetEntries()):
        TNeighbours.GetEntry(i)
        cID = TNeighbours.cellId
        if cID == cellID:
            print_entry(i, True, False)
            return

    print("CellID not found")


# =========================================
# Print info about n random cells
# =========================================
def print_random(n=10):
    import random
    nEntries = TNeighbours.GetEntries()

    for _ in range(n):
        i = random.randint(0, nEntries-1)
        print_entry(i, True, False)


    
# ================================
# parse arguments
# ================================

parser = argparse.ArgumentParser(description="Print cell info")
parser.add_argument("--neighbours-file", default=defaultFilenameNeighbours)
parser.add_argument("--noise", action="store_true",
                    help="Print noise info")
parser.add_argument("--noise-file", default=defaultFilenameNoise)
parser.add_argument("--read-xml", action="store_true",
                    help="Read encoding from compact file")
parser.add_argument("--xml-file", default=defaultCompactFile)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--cells",
                   help="Comma-separated list of cellIDs")
group.add_argument("--random", type=int,
                   help="Print N random entries")
args = parser.parse_args()
filenameNeighbours = args.neighbours_file
filenameNoise = args.noise_file
compactFile = args.xml_file


# ================================
# Load encoders
# ================================
loadCoders(args.read_xml, compactFile)


# ================================
# Load trees
# ================================

fNeighbours = ROOT.TFile.Open(filenameNeighbours)
TNeighbours = fNeighbours.Get(treenameNeighbours)

neighbours = ROOT.std.vector('unsigned long')()
TNeighbours.SetBranchAddress("neighbours", neighbours)

fNoise = None
TNoise = None
if args.noise:
    fNoise = ROOT.TFile.Open(filenameNoise)
    TNoise = fNoise.Get(treenameNoise)


# ================================
# Print cell info
# ================================
if args.cells:
    cell_list = [int(x) for x in args.cells.split(",")]
    print_cells(cell_list)
elif args.random:
    print_random(args.random)
else:
    print("ERROR: specify --cells or --random")
