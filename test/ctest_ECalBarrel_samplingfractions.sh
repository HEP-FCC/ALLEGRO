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
--steeringFile "${FCCCONFIG}/FullSim/ALLEGRO/ALLEGRO_o1_v03/SteeringFile_ALLEGRO_o1_v03.py"
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
# compare the sampling fractions with the reference ones: they are compatible if, in each layer,
# |SF - SF_ref| < N * sqrt(SFerr^2 + SFerr_ref^2), with N = SF_NSIGMA (default: 3)
# (SF_NSIGMA=0 requires the sampling fractions to be exactly equal to the reference ones)
# return 0 if compatible and 1 if not; a * marks the layers that are not compatible
# note that this requirement is pretty conservative given that the generated events are the
# same (same random generator's seed number)
SF_NSIGMA=${SF_NSIGMA:-3}
set +e
python - "SF_ecalbarrel.json" "${SCRIPT_DIR}/SF_ref_ecalbarrel.json" "${SF_NSIGMA}" <<'EOF'
import json
import math
import sys

with open(sys.argv[1]) as f:
    sf_new = json.load(f)
with open(sys.argv[2]) as f:
    sf_ref = json.load(f)
nsigma = float(sys.argv[3])

compatible = len(sf_new["SF"]) == len(sf_ref["SF"])
if not compatible:
    print(f"Number of layers differs: {len(sf_new['SF'])} (produced) vs {len(sf_ref['SF'])} (reference)")

print(f"{'layer':>5} {'SF':>10} {'SFerr':>10} {'SF ref':>10} {'SFerr ref':>10} {'rel. diff':>10} {'pull':>7}")
for layer, (sf, err, sf0, err0) in enumerate(zip(sf_new["SF"], sf_new["SFerr"], sf_ref["SF"], sf_ref["SFerr"])):
    diff = sf - sf0
    rel_diff = diff / sf0 if sf0 != 0 else float("inf")
    sigma = math.sqrt(err**2 + err0**2)
    pull = diff / sigma if sigma > 0 else (0. if diff == 0 else float("inf"))
    if nsigma > 0:
        ok = abs(pull) < nsigma
    else:
        ok = diff == 0
    if not ok:
        compatible = False
    print(f"{layer:>5} {sf:>10.6f} {err:>10.6f} {sf0:>10.6f} {err0:>10.6f} {rel_diff:>10.2e} {pull:>7.2f}" + ("" if ok else " *"))

criterion = f"within {nsigma:g} sigma" if nsigma > 0 else "exactly equal"
if compatible:
    print(f"The sampling fractions are compatible with the reference ones ({criterion})")
    sys.exit(0)
print(f"The sampling fractions are NOT compatible with the reference ones ({criterion})")
sys.exit(1)
EOF
SF_STATUS=$?
set -e
if [ ${SF_STATUS} -ne 0 ]; then
    exit 1
fi

echo "Completed successfully!"
exit 0
