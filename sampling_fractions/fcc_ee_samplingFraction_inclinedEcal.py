# Input settings
momentum = 10  # Particle gun momentum (in GeV) used for G4 generation
from os import environ
path_to_detector = environ.get("K4GEO", "")
detectors_to_use = [
    'FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/DectEmptyMaster.xml',
    'FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ECalBarrel_thetamodulemerged_calibration.xml'
]
ecalBarrelReadoutName = "ECalBarrelModuleThetaMerged"
ecalBarrelLayers = 11


# Main program

# from Configurables import EventCounter
from Gaudi.Configuration import INFO

# Input/Output handling
from k4FWCore import IOSvc
from Configurables import EventDataSvc
io_svc = IOSvc("IOSvc")
io_svc.Input = "ALLEGRO_calibration_sim.root"
io_svc.Output = "fccee_samplingFraction_inclinedEcal.root"
podioevent = EventDataSvc("EventDataSvc")

# DD4hep geometry service
from Configurables import GeoSvc
from os import path
geoservice = GeoSvc("GeoSvc",
                    OutputLevel=INFO)
geoservice.detectors = [path.join(
    path_to_detector, _det) for _det in detectors_to_use]

# algorithm that calculates the sampling fractions
from Configurables import SamplingFractionInLayers
hist = SamplingFractionInLayers("hists",
                                energyAxis=momentum,
                                readoutName=ecalBarrelReadoutName,
                                layerFieldName="layer",
                                activeFieldName="type",
                                activeFieldValue=0,
                                numLayers=ecalBarrelLayers,
                                OutputLevel=INFO)
hist.deposits.Path = "ECalBarrelModuleThetaMerged"

# save the histrograms with the sampling fractions
from Configurables import THistSvc
THistSvc().Output = [
    "rec DATAFILE='calibration_output_pdgID_11_pMin_10000_pMax_10000_thetaMin_55_thetaMax_125_ddsim.root' TYP='ROOT' OPT='RECREATE'"]  # FIXME this should better be set based on the values used to create the G4 file
THistSvc().PrintAll = True
THistSvc().AutoSave = True
THistSvc().AutoFlush = False
THistSvc().OutputLevel = INFO

# CPU information
from Configurables import AuditorSvc
from Configurables import ChronoAuditor
chra = ChronoAuditor()
audsvc = AuditorSvc()
audsvc.Auditors = [chra]
hist.AuditExecute = True

# PODIO algorithm
io_svc.outputCommands = ["drop *"]

# event_counter = EventCounter('event_counter')
# event_counter.Frequency = 10

# ApplicationMgr
from k4FWCore import ApplicationMgr
ApplicationMgr(
    TopAlg=[
        #event_counter,
        hist],
    EvtSel='NONE',
    EvtMax=-1,
    # order is important, as GeoSvc is needed by G4SimSvc
    ExtSvc=[geoservice, podioevent, audsvc],
    OutputLevel=INFO,
    StopOnSignal=True
)
