# ecalB:
#                                        includeDiagonalCells=False,
#                                        includeDiagonalCellsHCal=False,
#                                        connectBarrels=False,
#                                        connectHCal=False,
#                                        connectECal=False,
# ecalE:
#                                        includeDiagonalCells=True,
#                                        includeDiagonalCellsHCal=False,
#                                        connectBarrels=False,
#                                        connectHCal=False,
#                                        connectECal=False,
#
# hcal: do we need this?
#                                            activeVolumesTheta=[
#                                                [],
#                                                [
#                                                    0.788969, 0.797785, 0.806444, 0.814950, 0.823304,
#                                                    0.839573, 0.855273, 0.870425, 0.885051, 0.899172,
#                                                    0.912809, 0.938708, 0.962896
#                                                ]
#                                            ],

#
# parse command line options
#
from k4FWCore.parseArgs import parser
parser.add_argument("--detector", type=str, default="FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml", help="The detector xml file")
parser.add_argument("--ecalb", action="store_true", help="produce map for ECal barrel")
parser.add_argument("--ecalec", action="store_true", help="produce map for ECal endcap")
parser.add_argument("--hcalb", action="store_true", help="produce map for HCal barrel")
parser.add_argument("--hcalec", action="store_true", help="produce map for HCal endcap")
parser.add_argument("--link-calos", action="store_true", help="link ECAL+HCAL")
parser.add_argument("--link-ecal", action="store_true", help="link ECAL barrel+endcap")
parser.add_argument("--link-hcal", action="store_true", help="link HCAL barrel+endcap")
parser.add_argument("--diagonal-ecal", action="store_true", help="include diagonal cells for ECal")
parser.add_argument("--diagonal-hcal", action="store_true", help="include diagonal cells for HCal")
parser.add_argument("--debug", action="store_true", help="turn on debug output")

myopts = parser.parse_known_args()[0]
doECalB = myopts.ecalb
doECalEC = myopts.ecalec
doHCalB = myopts.hcalb
doHCalEC = myopts.hcalec
connectBarrels = myopts.link_calos
connectECal = myopts.link_ecal
connectHCal = myopts.link_hcal
includeDiagonalCells = myopts.diagonal_ecal
includeDiagonalCellsHCal = myopts.diagonal_hcal
debug = myopts.debug

#
# some constants
#
ecalBarrelNumLayers = 11
ecalEndCapNumLayers = 98
hcalBarrelNumLayers = 13
hcalEndCapNumLayers = 37

#
# create lists of parameters needed later
#
readoutNames = []
systemNames = []
systemValues = []
activeFieldNames = []
activeVolumesNumbers = []
activeVolumesTheta = []
outputFileName = "neighbours_map"

if doECalB:
    print("including ECAL barrel")
    readoutNames.append("ECalBarrelModuleThetaMerged")
    systemNames.append("system")
    systemValues.append(4)
    activeFieldNames.append("layer")
    activeVolumesNumbers.append(ecalBarrelNumLayers)
    activeVolumesTheta.append([])
    outputFileName += "_ecalB"

if doECalEC:
    print("including ECAL endcap")
    readoutNames.append("ECalEndcapTurbine")
    systemNames.append("system")
    systemValues.append(5)
    activeFieldNames.append("layer")
    activeVolumesNumbers.append(ecalEndCapNumLayers)
    activeVolumesTheta.append([])
    outputFileName += "_ecalE"

if doHCalB:
    print("including HCAL barrel")
    readoutNames.append("HCalBarrelReadout")
    systemNames.append("system")
    systemValues.append(8)
    activeFieldNames.append("layer")
    activeVolumesNumbers.append(hcalBarrelNumLayers)
    activeVolumesTheta.append([])
    outputFileName += "_hcalB"

if doHCalEC:
    print("including HCAL endcap")
    readoutNames.append("HCalEndcapReadout")
    systemNames.append("system")
    systemValues.append(9)
    activeFieldNames.append("layer")
    activeVolumesNumbers.append(hcalEndCapNumLayers)
    activeVolumesTheta.append([])
    outputFileName += "_hcalE"

outputFileName += ".root"
print("map will be saved to file", outputFileName)

#
# create map
#
from Gaudi.Configuration import INFO, DEBUG


# Detector geometry
# prefix all xmls with path_to_detector
# if K4GEO is empty, this should use relative path to working directory
from Configurables import GeoSvc
import os
geoservice = GeoSvc("GeoSvc")
path_to_detector = os.environ.get("K4GEO", "")
print("reading detector compact files from", path_to_detector)
detectors_to_use = [
    myopts.detector
]
geoservice.detectors = [os.path.join(
    path_to_detector, _det) for _det in detectors_to_use]
# geoservice.OutputLevel = DEBUG
geoservice.OutputLevel = INFO


from Configurables import CreateFCCeeCaloNeighbours
neighbours = CreateFCCeeCaloNeighbours("neighbours",
                                       outputFileName=outputFileName,
                                       readoutNames=readoutNames,
                                       systemNames=systemNames,
                                       systemValues=systemValues,
                                       activeFieldNames=activeFieldNames,
                                       activeVolumesNumbers=activeVolumesNumbers,
                                       activeVolumesTheta=activeVolumesTheta,
                                       includeDiagonalCells=includeDiagonalCells,
                                       includeDiagonalCellsHCal=includeDiagonalCellsHCal,
                                       connectBarrels=connectBarrels,
                                       connectECal=connectECal,
                                       connectHCal=connectHCal)
if debug:
    neighbours.OutputLevel=DEBUG
else:
    neighbours.OutputLevel=INFO


# configure the application
from k4FWCore import ApplicationMgr
ApplicationMgr(TopAlg=[],
               EvtSel='NONE',
               EvtMax=1,
               # order is important, as GeoSvc is needed by G4SimSvc
               ExtSvc=[geoservice, neighbours],
               OutputLevel=INFO
               )
