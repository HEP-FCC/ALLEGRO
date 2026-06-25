from k4FWCore.parseArgs import parser
parser.add_argument("--detector", type=str, default="FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml", help="The detector xml file")
parser.add_argument("--subdetectors", type=str, nargs="+", default=["ecalb", "ecale", "hcalb", "hcale"],
                    help="List of subdetectors: ecalb, ecale, hcalb, hcale"
)


args = parser.parse_known_args()

subdetector_map = {
    "ecalb": "ecalBarrel",
    "ecale": "ecalEndcap",
    "hcalb": "hcalBarrel",
    "hcale": "hcalEndcap",
}

invalid = [s for s in args[0].subdetectors if s not in subdetector_map]
if invalid:
    raise ValueError(f"Unknown subdetectors: {invalid}")
detectors = [subdetector_map[s] for s in args[0].subdetectors]

ecalBarrelNumLayers = 11
ecalEndcapNumLayers = 98
ecalEndcapNumWheels = 3
hcalBarrelNumLayers = 13
hcalEndcapNumLayers = 37

ecalBarrelSysId = 4
ecalEndcapSysId = 5
hcalBarrelSysId = 8
hcalEndcapSysId = 9

from Gaudi.Configuration import INFO, DEBUG

# Detector geometry
# prefix all xmls with path_to_detector
# if K4GEO is empty, this should use relative path to working directory
from Configurables import GeoSvc
import os
geoservice = GeoSvc("GeoSvc")
path_to_detector = os.environ.get("K4GEO", "")
compactFile = args[0].detector
full_path = os.path.join(path_to_detector, compactFile)
import sys
if not os.path.isfile(full_path):
    print(f"Error: compact file '{full_path}' does not exist.", file=sys.stderr)
    sys.exit(1)

geoservice.detectors = [full_path]
geoservice.OutputLevel = INFO

# readout names for ECAL and HCAL
ecalBarrelReadoutName = "ECalBarrelModuleThetaMerged"
ecalEndcapReadoutName = "ECalEndcapTurbine"
hcalBarrelReadoutName = "HCalBarrelReadout"
hcalEndcapReadoutName = "HCalEndcapReadout"

readoutNames = []
systemNames = []
systemValues = []
activeFieldNames = []
activeVolumesNumbers = []
activeVolumesTheta = []
outputFileName = "cellNoise_map_electronicsNoiseLevel"

if "ecalBarrel" in detectors:
    ecalBarrelNoiseFile = "noise_capa_ecalbarrel/elecNoise_ecalBarrelFCCee_theta.root"
    ecalBarrelNoiseRMSHistName = "h_elecNoise_fcc_"
    if not os.path.isfile(ecalBarrelNoiseFile):
        print(f"Error: noise file '{ecalBarrelNoiseFile}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # cell positioning and noise tool for the ecal barrel
    from Configurables import CellPositionsECalBarrelModuleThetaSegTool
    ECalBarrelPositionsTool = CellPositionsECalBarrelModuleThetaSegTool("CellPositionsECalBarrel",
                                                                        readoutName=ecalBarrelReadoutName)

    from Configurables import NoiseCaloCellsFromFileBarrelTool
    ECalBarrelNoiseTool = NoiseCaloCellsFromFileBarrelTool("NoiseCaloCellsFromFileBarrelTool",
                                                           cellPositionsTool=ECalBarrelPositionsTool,
                                                           readoutName=ecalBarrelReadoutName,
                                                           noiseFileName=ecalBarrelNoiseFile,
                                                           elecNoiseRMSHistoName=ecalBarrelNoiseRMSHistName,
                                                           setNoiseOffset=False,
                                                           activeFieldName="layer",
                                                           addPileup=False,
                                                           numHistograms=ecalBarrelNumLayers,
                                                           scaleFactor=1 / 1000.,  # MeV to GeV
                                                           OutputLevel=INFO)

    readoutNames += [ecalBarrelReadoutName]
    systemNames += ["system"]
    systemValues += [ecalBarrelSysId]
    activeFieldNames += ["layer"]
    activeVolumesNumbers += [ecalBarrelNumLayers]
    activeVolumesTheta += [[]]
    outputFileName = outputFileName + "_ecalB_" + ecalBarrelReadoutName
else:
    ECalBarrelNoiseTool = None

if "ecalEndcap" in detectors:
    ecalEndcapNoiseFile =    "./noise_capa_ecalendcap/elecNoise_ecalendcap.root"
    if not os.path.isfile(ecalEndcapNoiseFile):
        print(f"Error: noise file '{ecalEndcapNoiseFile}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # cell positioning and noise tool for the ecal endcap
    from Configurables import CellPositionsECalEndcapTurbineSegTool
    ECalEndcapPositionsTool = CellPositionsECalEndcapTurbineSegTool("CellPositionsECalEndcap",
                                                                    readoutName=ecalEndcapReadoutName)

    from Configurables import NoiseCaloCellsFromFileTurbineEndcapTool
    ECalEndcapNoiseTool = NoiseCaloCellsFromFileTurbineEndcapTool("NoiseCaloCellsFromFileTurbineEndcapTool",
                                                                  cellPositionsTool=ECalEndcapPositionsTool,
                                                                  readoutName=ecalEndcapReadoutName,
                                                                  noiseFileName=ecalEndcapNoiseFile,
                                                                  elecNoiseRMSHistoName="noise_endcap_wheel",
                                                                  setNoiseOffset=False,
                                                                  activeFieldName="wheel",
                                                                  addPileup=False,
                                                                  numHistograms=ecalEndcapNumWheels,
                                                                  scaleFactor = 1/1000., #MeV to GeV
                                                                  OutputLevel=INFO)

    readoutNames += [ecalEndcapReadoutName]
    systemNames += ["system"]
    systemValues += [ecalEndcapSysId]
    activeFieldNames += ["layer"]
    activeVolumesNumbers += [ecalEndcapNumLayers]
    activeVolumesTheta += [[]]
    outputFileName = outputFileName + "_ecalE_" + ecalEndcapReadoutName

else:
    ECalEndcapNoiseTool = None

if "hcalBarrel" in detectors:
    # noise tool for the HCAL barrel
    # HCAL noise file has yet to be created/implemented
    # HCalNoiseTool = ReadNoiseFromFileTool("ReadNoiseFromFileToolHCal",
    #                                       readoutName = hcalBarrelReadoutName,
    #                                       noiseFileName = BarrelNoisePath,
    #                                       elecNoiseHistoName = ecalBarrelNoiseHistName,
    #                                       setNoiseOffset = False,
    #                                       activeFieldName = "layer",
    #                                       addPileup = False,
    #                                       numRadialLayers = 12,
    #                                       scaleFactor = 1/1000., #MeV to GeV
    #                                       OutputLevel = INFO)
    # ConstNoiseTool provides constant noise for all calo subsystems
    # here we are going to use it only for hcal barrel
    from Configurables import ConstNoiseTool
    HCalBarrelNoiseTool = ConstNoiseTool("HCalBarrelNoiseTool",
                                         detectors=["HCAL_Barrel"],
                                         detectorsNoiseRMS=[0.0115 / 4],
                                         OutputLevel=DEBUG)
    readoutNames += [hcalBarrelReadoutName]
    systemNames += ["system"]
    systemValues += [hcalBarrelSysId]
    activeFieldNames += ["layer"]
    activeVolumesNumbers += [hcalBarrelNumLayers]
    activeVolumesTheta += [
    #    [0.788969, 0.797785, 0.806444, 0.814950, 0.823304,
    #     0.839573, 0.855273, 0.870425, 0.885051, 0.899172,
    #     0.912809, 0.938708, 0.962896]
        []
    ]
    outputFileName = outputFileName + "_hcalB_" + hcalBarrelReadoutName
else:
    HCalBarrelNoiseTool = None

if "hcalEndcap" in detectors:
    from Configurables import ConstNoiseTool
    HCalEndcapNoiseTool = ConstNoiseTool("HCalEndcapNoiseTool",
                                         detectors=["HCAL_Endcap"],
                                         detectorsNoiseRMS=[0.0115 / 4],
                                         OutputLevel=DEBUG)
    readoutNames += [hcalEndcapReadoutName]
    systemNames += ["system"]
    systemValues += [hcalEndcapSysId]
    activeFieldNames += ["layer"]
    activeVolumesNumbers += [hcalEndcapNumLayers]
    activeVolumesTheta += [
        []
    ]
    outputFileName = outputFileName + "_hcalE_" + hcalEndcapReadoutName
else:
    HCalEndcapNoiseTool = None


# create the noise file
# the tool wants the system IDs, maybe we could have passed the names instead
outputFileName += ".root"
from Configurables import CreateFCCeeCaloNoiseLevelMap
noisePerCell = CreateFCCeeCaloNoiseLevelMap("noisePerCell",
                                            ECalBarrelNoiseTool=ECalBarrelNoiseTool,
                                            ecalBarrelSysId=ecalBarrelSysId,
                                            ECalEndcapNoiseTool=ECalEndcapNoiseTool,
                                            ecalEndcapSysId=ecalEndcapSysId,
                                            HCalBarrelNoiseTool=HCalBarrelNoiseTool,
                                            hcalBarrelSysId=hcalBarrelSysId,
                                            HCalEndcapNoiseTool=HCalEndcapNoiseTool,
                                            hcalEndcapSysId=hcalEndcapSysId,
                                            readoutNames=readoutNames,
                                            systemNames=systemNames,
                                            systemValues=systemValues,
                                            activeFieldNames=activeFieldNames,
                                            activeVolumesNumbers=activeVolumesNumbers,
                                            activeVolumesTheta=activeVolumesTheta,
                                            outputFileName=outputFileName,
                                            OutputLevel=DEBUG)

# configure the application
from Configurables import ApplicationMgr
ApplicationMgr(TopAlg=[],
               EvtSel='NONE',
               EvtMax=1,
               # order is important, as GeoSvc is needed by G4SimSvc
               ExtSvc=[geoservice, noisePerCell, 'RndmGenSvc'],
               OutputLevel=INFO
               )
print("Noise map will be saved in file", noisePerCell.outputFileName)
