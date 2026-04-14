from Gaudi.Configuration import *
import os

# Detector geometry
from Configurables import GeoSvc
geoservice = GeoSvc("GeoSvc")
# if FCC_DETECTORS is empty, this should use relative path to working directory
path_to_detector = os.environ.get("FCCDETECTORS", "")
print(path_to_detector)
detectors_to_use=[
                    './FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml',
                  ]
# prefix all xmls with path_to_detector
geoservice.detectors = [os.path.join(path_to_detector, _det) for _det in detectors_to_use]
geoservice.OutputLevel = INFO

from Configurables import CreateFCCeeCaloNoiseLevelMap
from Configurables import NoiseCaloCellsTurbineEndcapFromFileTool

endcapNoiseFile =    "../ALLEGRO/noise_maps/noise_capa_ecalendcap/elecNoise_ecalendcap.root"

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
