#!/bin/bash
set -e

# --- Key4hep Setup ---
if [[ -z "${KEY4HEP_STACK}" ]]; then
    echo "Sourcing Key4hep environment..."
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
else
    echo "The Key4hep stack is already loaded."
fi

# Workaround to have ctests working (get the directory of this script)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# particle energy in GeV
ENERGY=20

# number of events
NEVENTS=100

# --- Build ddsim command ---
DDSIM_CMD=(
ddsim
--outputFile "ALLEGRO_sf_sim.root"
--compactFile "${K4GEO}/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/DectEmptyMaster.xml" "${K4GEO}/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ECalBarrel_thetamodulemerged_calibration.xml"
--steeringFile "${FCCCONFIG}/FCCee/FullSim/ALLEGRO/ALLEGRO_o1_v03/SteeringFile_ALLEGRO_o1_v03.py"
--numberOfEvents "${NEVENTS}"
--random.enableEventSeed
--random.seed 42
--enableGun
--gun.distribution uniform
--gun.energy "${ENERGY}*GeV"
--gun.thetaMin "90*degree"
--gun.thetaMax "90*degree"
--gun.particle "e-"
--crossingAngleBoost 0.0
)

# Run the SIM step
echo "Running: ${DDSIM_CMD[*]}"
"${DDSIM_CMD[@]}"

# --- ANALYSIS Step ---
DIGI_CMD=(
k4run "../../../sampling_fractions/fcc_ee_samplingFraction_inclinedEcal.py"
--IOSvc.Input "ALLEGRO_sf_sim.root"
--IOSvc.Output "ALLEGRO_sf_k4run.root"
-n "${NEVENTS}"
--hists.energyAxis "${ENERGY}"
--Output.THistSvc "rec DATAFILE='ALLEGRO_sf_calib_e-_energy_${ENERGY}GeV_theta_90.root' TYP='ROOT' OPT='RECREATE'"
)

echo "Running: ${DIGI_CMD[*]}"
"${DIGI_CMD[@]}"

# --- PLOTTING and EXTRACTION OF RESULTS
python ../../../sampling_fractions/FCC_calo_analysis_cpp/plot_samplingFraction.py \
       "ALLEGRO_sf_calib_e-_energy_${ENERGY}GeV_theta_90.root" "$ENERGY" \
       --preview -outputfolder "." --noFits --json SF_ecalbarrel.json

# --- COMPARISON TO REFERENCE
# compare the sampling fractions with the reference ones, return 0 if equal and 1 if not
# (a * marks the layers whose SF differ from the reference)
set +e
python - "SF_ecalbarrel.json" "${SCRIPT_DIR}/SF_ref_ecalbarrel.json" <<'EOF'
import json
import sys

with open(sys.argv[1]) as f:
    sf = json.load(f)["SF"]
with open(sys.argv[2]) as f:
    ref = json.load(f)
sf_ref = ref["SF"]
sf_err_ref = ref["SFerr"]

equal = len(sf) == len(sf_ref)
if not equal:
    print(f"Number of layers differs: {len(sf)} (produced) vs {len(sf_ref)} (reference)")

print(f"{'layer':>5} {'SF':>12} {'SF ref':>12} {'rel. diff':>12} {'SFerr ref':>12}")
for layer, (value, value_ref, err_ref) in enumerate(zip(sf, sf_ref, sf_err_ref)):
    rel_diff = (value - value_ref) / value_ref if value_ref != 0 else float("inf")
    differs = value != value_ref
    if differs:
        equal = False
    print(f"{layer:>5} {value:>12.6f} {value_ref:>12.6f} {rel_diff:>12.3e} {err_ref:>12.6f}" + (" *" if differs else ""))

if equal:
    print("The sampling fractions are equal to the reference ones")
    sys.exit(0)
print("The sampling fractions are NOT equal to the reference ones")
sys.exit(1)
EOF
SF_STATUS=$?
set -e
if [ ${SF_STATUS} -ne 0 ]; then
    exit 1
fi

echo "Completed successfully!"
exit 0
