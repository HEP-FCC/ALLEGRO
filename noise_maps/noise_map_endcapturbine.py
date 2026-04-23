from Gaudi.Configuration import *
import os
import sys

from k4FWCore.parseArgs import parser
parser.add_argument("--compactFile", type=str, help="Top-level compact file name", required=True)
opts = parser.parse_known_args()[0]
compactFile = opts.compactFile

# Detector geometry
from Configurables import GeoSvc
geoservice = GeoSvc("GeoSvc")
# if K4GEO is empty, this should use relative path to working directory
path_to_detector = os.environ.get("K4GEO", "")
print(path_to_detector)
full_path = os.path.join(path_to_detector, compactFile)
if not os.path.isfile(full_path):
    print(f"Error: compact file '{full_path}' does not exist.", file=sys.stderr)
    sys.exit(1)
    
# prefix all xmls with path_to_detector
geoservice.detectors = [full_path]
geoservice.OutputLevel = INFO

from Configurables import CreateFCCeeCaloNoiseLevelMap
from Configurables import NoiseCaloCellsTurbineEndcapFromFileTool

endcapNoiseFile =    "./noise_capa_ecalendcap/elecNoise_ecalendcap.root"

eCalEndcapNoiseTool = NoiseCaloCellsTurbineEndcapFromFileTool("NoiseCaloCellsTurbineEndcapFromFileTool",
                                                              noiseFileName=endcapNoiseFile,
                                                              addPileup=False,
                                                              scaleFactor = 1/1000., #MeV to GeV
                                                              elecNoiseRMSHistoName="noise_endcap_wheel")

noisePerCell = CreateFCCeeCaloNoiseLevelMap("noisePerCell",
                                            ECalBarrelNoiseTool=None,
                                            ECalEndcapNoiseTool=eCalEndcapNoiseTool,
                                            HCalBarrelNoiseTool=None,
                                            HCalEndcapNoiseTool=None,
                                            systemValues=[5],
                                            outputFileName="cellNoise_map_endcapTurbine_electronicsNoiseLevel.root",
                                            readoutNames = ["ECalEndcapTurbine"],
                                            OutputLevel=INFO)


# ApplicationMgr
from Configurables import ApplicationMgr
ApplicationMgr( TopAlg = [],
                EvtSel = 'NONE',
                EvtMax   = 1,
                # order is important, as GeoSvc is needed by G4SimSvc
                ExtSvc = [geoservice, noisePerCell,"RndmGenSvc"],
                OutputLevel=INFO
)
