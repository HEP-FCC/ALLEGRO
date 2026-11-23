#!/bin/bash

# set-up the Key4hep environment if not already set
if [[ -z "${KEY4HEP_STACK}" ]]; then
  source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
else
  echo "The Key4hep stack was already loaded in this environment."
fi


if [ -z "${ALLEGRO+x}" ]; then
    ALLEGRO=../
fi


# write the example configuration file
echo
echo "##########################################"
echo "# Writing the example configuration file #"
echo "##########################################"
echo
configFile=soup.gun
rm -f $configFile
python $ALLEGRO/utils/generate_multiparticle_gun.py --write-example $configFile || exit 1
if [ ! -f $configFile ]; then
    echo "Configuration file missing"
    exit -1
fi

# generate 100 events (not simulated), twice with the same seed
echo
echo "#############################"
echo "# Generating 100 events     #"
echo "#############################"
echo
outFile=soup.hepmc3
rm -f $outFile soup_again.hepmc3
python $ALLEGRO/utils/generate_multiparticle_gun.py --config $configFile -n 100 --seed 1 -o $outFile || exit 1
python $ALLEGRO/utils/generate_multiparticle_gun.py --config $configFile -n 100 --seed 1 -o soup_again.hepmc3 || exit 1
if [ ! -f $outFile ]; then
    echo "Output file missing"
    exit -1
fi

# check the events against the configuration of the example
echo
echo "#############################"
echo "# Checking the events       #"
echo "#############################"
echo
python - $outFile <<'PYEOF' || exit 1
import math
import sys
from pyHepMC3 import HepMC3 as hm

# configuration of the example (--write-example)
npart_range = (10, 15)
eta_range = (-0.5, 0.5)
mom_range = (0.5, 50)
drmax = 0.5
pids = {211, -211, 130, 2112, 22, 11, -11, 13, -13, 321, -321, 2212, -2212}

reader = hm.ReaderAscii(sys.argv[1])
nevents = 0
errors = []
while True:
    event = hm.GenEvent()
    reader.read_event(event)
    if reader.failed():
        break
    nevents += 1
    beams = [p for p in event.particles() if p.status() == 4]
    final = [p for p in event.particles() if p.status() == 1]
    if sorted(p.pid() for p in beams) != [-11, 11]:
        errors.append(f"event {nevents}: beam particles {[p.pid() for p in beams]}")
    if not npart_range[0] <= len(final) <= npart_range[1]:
        errors.append(f"event {nevents}: {len(final)} particles")
    etas = []
    for p in final:
        mom = p.momentum()
        pabs = math.sqrt(mom.px() ** 2 + mom.py() ** 2 + mom.pz() ** 2)
        if p.pid() not in pids:
            errors.append(f"event {nevents}: unexpected PDG code {p.pid()}")
        if not mom_range[0] <= pabs <= mom_range[1]:
            errors.append(f"event {nevents}: momentum {pabs} GeV")
        etas.append(mom.eta())
    # the particles are in a cone of radius drmax (in eta) around an axis within eta_range
    if etas and (max(etas) - min(etas) > 2 * drmax + 1e-6 or
                 min(etas) < eta_range[0] - drmax - 1e-6 or max(etas) > eta_range[1] + drmax + 1e-6):
        errors.append(f"event {nevents}: eta between {min(etas)} and {max(etas)}")
reader.close()
if nevents != 100:
    errors.append(f"{nevents} events instead of 100")
for e in errors[:20]:
    print("ERROR:", e)
if errors:
    sys.exit(1)
print(f"{nevents} events checked")
PYEOF

# the same seed must give the same events
if ! cmp -s $outFile soup_again.hepmc3; then
    echo "The two files generated with the same seed differ"
    exit -1
fi
echo "Same seed, same events"
rm -f soup_again.hepmc3

echo
echo "#############################"
echo "# SUCCESS                   #"
echo "#############################"
echo
