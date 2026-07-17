# Input settings
from os import environ
path_to_detector = environ.get("K4GEO", "")
detectors_to_use = [
    'FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/DectEmptyMaster.xml',
    'FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ECalBarrel_thetamodulemerged_upstream.xml'
]
ecalBarrelReadoutName = "ECalBarrelModuleThetaMerged"
ecalBarrelSamplingFractions = [0.3800493723322256] * 1 + [0.13494147915064658] * 1 + [0.142866851721152] * 1 + [0.14839315921940666] * 1 + [0.15298362570665006] * 1 + [0.15709704561942747] * 1 + [0.16063717490147533] * 1 + [0.1641723795419055] * 1 + [0.16845490287689746] * 1 + [0.17111520115997653] * 1 + [0.1730605163148862] * 1
ecalBarrelLayers = 11


# Main program

# from Configurables import EventCounter
from Gaudi.Configuration import INFO

# Input/Output handling
from k4FWCore import IOSvc
from Configurables import EventDataSvc
podioevent = EventDataSvc("EventDataSvc")
io_svc = IOSvc("IOSvc")
io_svc.Input = "ALLEGRO_calibration_sim.root"
io_svc.Output = "fccee_deadMaterial_inclinedEcal.root"

# DD4hep geometry service
from Configurables import GeoSvc
from os import path
geoservice = GeoSvc("GeoSvc",
                    OutputLevel=INFO)
geoservice.detectors = [path.join(
    path_to_detector, _det) for _det in detectors_to_use]


# algorithm that creates the cells
# not sure why its needed, probably only because of the type of objects needed by the EnergyInCaloLayers tool?
# I am asking because with ddsim the hits are already merged according to the readout granularity,
# and here there is no calibration added (its applied in the EnergyInCaloLayers tool) nor noise. Positions are set, but are not really relevant
# (only quantity used I think is cellID from which the layer is extracted, and cell energy)
from Configurables import CreateCaloCells
createcellsBarrel = CreateCaloCells("CreateCaloCellsBarrel",
                                    doCellCalibration=False,
                                    addPosition=True,
                                    addCellNoise=False,
                                    filterCellNoise=False)
createcellsBarrel.hits.Path = ecalBarrelReadoutName
createcellsBarrel.cells.Path = "ECalBarrelCells"

# algorithm that calculates the energy in the layers
from Configurables import EnergyInCaloLayers
energy_in_layers = EnergyInCaloLayers("energyInLayers",
                                      readoutName=ecalBarrelReadoutName,
                                      numLayers=ecalBarrelLayers,
                                      # sampling fraction is given as the energy correction will be applied on calibrated cells
                                      samplingFractions=ecalBarrelSamplingFractions,
                                      OutputLevel=INFO)
energy_in_layers.deposits.Path = createcellsBarrel.cells.Path
energy_in_layers.particle.Path = "MCParticles"

# CPU information
from Configurables import AuditorSvc
from Configurables import ChronoAuditor
chra = ChronoAuditor()
audsvc = AuditorSvc()
audsvc.Auditors = [chra]
energy_in_layers.AuditExecute = True

# Configure output
io_svc.outputCommands = ["keep *"]
io_svc.outputCommands.append("drop %s" % ecalBarrelReadoutName)
io_svc.outputCommands.append("drop %sContributions" % ecalBarrelReadoutName)
io_svc.outputCommands.append("drop %s" % createcellsBarrel.cells.Path)
io_svc.outputCommands.append("drop MCParticles")

# event_counter = EventCounter('event_counter')
# event_counter.Frequency = 10

# ApplicationMgr
from k4FWCore import ApplicationMgr
ApplicationMgr(
    TopAlg=[
        #event_counter,
        createcellsBarrel,
        energy_in_layers,
    ],
    EvtSel='NONE',
    EvtMax=-1,
    # order is important, as GeoSvc is needed by G4SimSvc
    ExtSvc=[geoservice, podioevent, audsvc],
    OutputLevel=INFO,
    StopOnSignal=True
)
