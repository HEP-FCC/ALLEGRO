import ROOT
import dd4hep
import os
import sys

systemEB = 4
systemEEC = 5
systemHB = 8
systemHEC = 9

# maps for caching
readoutMap = {}
segMap = {}
encodingMap = {}
coderMap = {}

# ------------------------------------------------------------------
# Return readout name for given system
# ------------------------------------------------------------------
def readoutStr(system):
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
        readoutName = ""
    return readoutName

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
volman = detector.volumeManager()

 
# ------------------------------------------------------------------
# Read encoding maps before loop, so that INFO messages from
# segmentation classes do not pollute output later
# ------------------------------------------------------------------
systems = [systemEB, systemEEC, systemHB, systemHEC]
for system in systems:
    readoutMap[system] = detector.readout(readoutStr(system))
    segMap[system] = readoutMap[system].segmentation()
    encodingMap[system] = readoutMap[system].idSpec().fieldDescription()
    coderMap[system] = ROOT.dd4hep.BitFieldCoder(encodingMap[system])
    # to initialise layer info for Ecal barrel now
    if (system==systemEB):
        volumeID = system
        _ = segMap[system].position(volumeID)
        

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
    readoutName = readoutStr(system)
    if readoutName == "": continue

    # Get readout (use caching)
    if readoutName != previous_readout:
        if system in readoutMap:
            readout = readoutMap[system]
            seg = segMap[system]
            encoding = encodingMap[system]
            coder = coderMap[system]
        else:
            readout = detector.readout(readoutName)
            seg = readout.segmentation()
            encoding = readout.idSpec().fieldDescription()
            coder = ROOT.dd4hep.BitFieldCoder(encoding)
            readoutMap[system] = readout
            segMap[system] = seg
            encodingMap[system] = encoding
            coderMap[system] = coder
        previous_readout = readoutName

    print("System:", system)
    print("Readout:", readoutName)
    print("CellID encoding:", encoding)
    if (system!=systemEB):
        position = seg.position(cellID)
    else:
        volumeID = seg.volumeID(cellID);
        # print("volumeID:", volumeID)
        vc = volman.lookupContext(volumeID);
        inSeg = seg.position(cellID);
        outSeg = vc.localToWorld(inSeg);
        position = outSeg
    print(f"Position (rho/theta/phi): {position.rho()}, {position.theta()}, {position.phi()}")
    print(f"Position (rho/z/phi): {position.rho()}, {position.z()}, {position.phi()}")
    # Decode and print all fields
    for field in coder.fields():
        name = field.name()
        value = coder.get(cellID, name)
        print(f"{name}: {value}")


