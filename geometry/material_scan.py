from Configurables import MaterialScan
from Configurables import GeoSvc
import os
from Gaudi.Configuration import INFO

from Configurables import ApplicationMgr
ApplicationMgr().EvtSel = 'None'
ApplicationMgr().EvtMax = 1
ApplicationMgr().OutputLevel = INFO

from k4FWCore.parseArgs import parser
defaultECalBarrelXML="ECalBarrel_thetamodulemerged.xml"
parser.add_argument("--trackeronly", action="store_true", help="Do material scan only for tracker", default=False)
parser.add_argument("--ecalonly", action="store_true", help="Do material scan only for ecal", default=False)
parser.add_argument("--ecalbarrelXML", default=defaultECalBarrelXML, type=str)

opts = parser.parse_known_args()[0]
trackeronly = opts.trackeronly
ecalonly = opts.ecalonly
if trackeronly and ecalonly:
    print("ERROR: trackeronly and ecalonly cannot be set simultaneously")
    exit
ecalbarrelXML = "FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/" + opts.ecalbarrelXML

# DD4hep geometry service
# parse the given xml file
path_to_detectors = os.environ.get("K4GEO", "")
geoservice = GeoSvc("GeoSvc")
if trackeronly:
    detcards = [
        'FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/DectEmptyMaster.xml',
        'FCCee/IDEA/compact/IDEA_o1_v04/VertexComplete_o1_v04.xml',
        'FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/DriftChamber_o1_v02.xml',
        'FCCee/IDEA/compact/IDEA_o1_v04/SiliconWrapper_o1_v02.xml',
    ]
elif ecalonly:
    detcards = [
        'FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/DectEmptyMaster.xml',
        ecalbarrelXML,
        'FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ECalEndcaps_Turbine_o1_v03.xml',
    ]
else:
    detcards = ['FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml']
geoservice.detectors = [
    os.path.join(path_to_detectors, detcard) for detcard in detcards
]
geoservice.OutputLevel = INFO
ApplicationMgr().ExtSvc += [geoservice]

# Material scan is done from the interaction point to the end of world volume.
# In order to use other end boundary, please provide the name of a thin, e.g. cylindrical volume.
# For instance adding envelopeName="BoundaryPostCalorimetry" will perform the scan only till the end of calorimetry.
# BoundaryPostCalorimetry is defined in Detector/DetFCChhECalInclined/compact/envelopePreCalo.xml
materialservice = MaterialScan("GeoDump")
materialservice.filename = "out_material_scan_#suffix.root"
# materialservice.etaBinning = 0.05
# materialservice.etaMax = 0.9
# full detector down to cos(theta)=0.9952 (theta = 7 degrees)
# materialservice.etaBinning = 0.1
# materialservice.etaMax = 2.8
# barrel down to cos(theta) = 0.766 (theta =  40 degrees)
# materialservice.etaBinning = 0.1
# materialservice.etaMax = 1.0
materialservice.etaBinning = float("#etabinning")
materialservice.etaMax = float("#etamax")
materialservice.nPhiTrials = 10
ApplicationMgr().ExtSvc += [materialservice]
