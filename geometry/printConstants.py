#
# print all constants and corresponding values in the compact files
# The user can decide which subdetectors or other elements to show or not
#
import ROOT
import dd4hep
import os
import sys

# elements to skip at parsing time
elementsToSkip = []

# elements to show in the end (skipped elements will not be shown)
elementsToShow=[
#    "mdi",
#    "vertex",
#    "dch",
#    "wrapper",
    "ecalb",
    "ecalec",
#    "hcal",
#    "lumical",
#    "muon",
#    "geom",
#    "dettype",
#    "readout",
#    "bfield"
    ]

def getElement(name):
    if name.startswith("BP"):
        return "mdi"
    elif "BeamPipe" in name:
        return "mdi"
    elif "beampipe" in name:
        return "mdi"
    elif name.startswith("QD0"):
        return "mdi"
    elif name.startswith("Collimator"):
        return "mdi"
    elif "HOMAbsorber" in name:
        return "mdi"
    elif name.startswith("BeamCal"):
        return "mdi"
    elif name.startswith("SeparatedBeamPipe"):
        return "mdi"
    elif name.startswith("SynchRadMaskSize"):
        return "mdi"
    elif name.startswith("MiddleOfSRMask_z"):
        return "mdi"
    elif name.startswith("CompSol"):
        return "mdi"
    elif name.startswith("Kicker"):
        return "mdi"
    elif name=="CrossingAngle":
        return "mdi"
    elif "Filler" in name:
        return "mdi"
    elif name[:-1]=="size_":
        return "mdi"
    elif name=="mask_epsilon":
        return "mdi"
    elif name=="env_safety":
        return "mdi"
    elif name.startswith("VTX"):
        return "vertex"
    elif "VXD" in name:
        return "vertex"
    elif "Vertex" in name:
        return "vertex"
    elif name.startswith("RSU"):
        return "vertex"
    elif "DCH" in name:
        return "dch"
    elif "SiWr" in name:
        return "wrapper"
    elif name.startswith("EMBarrel"):
        return "ecalb"
    elif name.startswith("CryoBarrel"):
        return "ecalb"
    elif name in ["safeMargin", "readout_thickness", "planeLength", "phi", "Steel_thickness", "Glue_thickness", "Pb_thickness", "Pb_thickness_max", "Sensitive_thickness", "AirMarginThickness", "InclinationAngle"]:
        return "ecalb"
    elif name.startswith("NLiqBathThickness"):
        return "ecalb"
    elif name.startswith("Bath_r"):
        return "ecalb"
    elif "ECAL_Barrel" in name:
        return "ecalb"
    elif "ECAL_Endcap" in name:
        return "ecalec"
    elif "ECalBarrel" in name:
        return "ecalb"
    elif "ECalEndcap" in name:
        return "ecalec"
    elif "EMEC" in name:
        return "ecalec"
    elif "Blade" in name:
        return "ecalec"
    elif name.startswith("nUnitCells"):
        return "ecalec"
    elif name.startswith("CryoEndcap"):
        return "ecalec"
    elif name.startswith("BathThickness"):
        return "ecalec"
    elif name in ["nWheels", "NobleLiquidGap"]:
        return "ecalec"
    elif "HCal" in name:
        return "hcal"
    elif "HCAL" in name:
        return "hcal"
    elif "LumiCal" in name:
        return "lumical"
    elif name.startswith("Lcal"):
        return "lumical"
    elif "Muon" in name:
        return "muon"
    elif name.startswith("world"):
        return "geom"
    elif name.startswith("compact_checksum"):
        return "geom"
    elif name.startswith("tracker_region"):
        return "geom"
    elif name.startswith("DetType"):
        return "dettype"
    elif "ReadoutID" in name:
        return "readout"
    elif name.startswith("Solenoid"):
        return "bfield"
    else:
        return "unknown"
    

# ------------------------------------------------------------------
# Load detector geometry
# ------------------------------------------------------------------
compactFile = "FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml"
path_to_detector = os.environ.get("K4GEO", "")
detectorFile = path_to_detector + "/" + compactFile
detector = dd4hep.Detector.getInstance()
detector.fromXML(detectorFile)
print("Loaded detector from compact file:", detectorFile)

print("")

# extract constants
constants = {}

for name, handle in detector.constants():
    # 1. Always get the string representation first
    pname = str(name)
    element = getElement(pname)
    
    
    raw_val = str(detector.constantAsString(name))
    constants[pname] = (element, raw_val)

for element in elementsToShow:
    if element in elementsToSkip:
        continue
    for name, value in constants.items():
        elem, raw_val = value
        if (elem!=element):
            continue
        try:
            # 2. Try to evaluate it as a double
            # This will catch things like "50.18*degree" or "10*mm"
            val_double = detector.constantAsDouble(name)
        
            # 3. Print the evaluated number
            print(f"{name:<50} | {element:<10} | {val_double:<20}")
        
        except (RuntimeError, Exception):
            # 4. If evaluation fails (like your Bitfield error), 
            # just print it as a raw string
            print(f"{name:<50} | {element:<10} | {raw_val:<20}")
