#!/usr/bin/env python3
#
# Particle gun with several particles per event, written as a HepMC3 ASCII file.
#
# Python reimplementation of the gun of the MLPF studies (D. Garcia, M. Selvaggi et al.),
# with the same configuration files and the same logic, plus a random seed
# (for reproducible productions) and an optional uniform cos(theta) sampling:
# - the number of particles is uniform in npart_range;
# - each particle is drawn uniformly from pid_list (repeat a PDG code to give it more weight); with exactly two
#   codes, they alternate;
# - the momentum is uniform in log(p) in mom_range [GeV];
# - with drmax > 0, an axis (eta, phi) is drawn for each event and the particles are drawn uniformly in a disc
#   of radius drmax around it in (eta, phi), to mimic a jet; with drmax = 0, each particle has its own direction;
# - directions: eta uniform in eta_range (as the original gun) or, if costheta_range is given, cos(theta) uniform
#   in costheta_range (better to cover the endcaps);
# - all the particles start at the origin, from a vertex with two dummy incoming beam particles (status 4).
#
# Configuration file (as for gun.cpp; lines starting with # are ignored):
#   npart_range 10,15
#   eta_range -0.5,0.5          (or: costheta_range -0.98,0.98)
#   mom_range 0.5,50
#   drmax 0.5
#   pid_list 211,-211,130,2112,22,11,-11,13,-13,321,-321,2212,-2212
#   nevents 100                 (overridden by -n)
# The values can also be given or changed with --set key=value (e.g. --set drmax=0.25).
#
# Needs the Key4hep environment (pyHepMC3). Example:
#   python generate_multiparticle_gun.py --config soup.gun -n 1000 --seed 1 -o soup.hepmc3
# An example configuration file, with comments, is written with:
#   python generate_multiparticle_gun.py --write-example [soup.gun]

import argparse
import math
import random


# masses [GeV] of the particles of the original gun (from Geant4 there), and of a few more
MASSES = {
    11: 0.000511, 13: 0.105658, 15: 1.77686, 22: 0.0, 111: 0.134977, 130: 0.497611, 310: 0.497611, 211: 0.139570,
    213: 0.77526, 321: 0.493677, 2112: 0.939565, 2212: 0.938272, 3112: 1.197449, 3122: 1.115683, 3222: 1.18937,
    3312: 1.32171,
}  # fmt: skip


def read_config(path, overrides):
    config = {}
    if path:
        with open(path) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                key, value = line.split(None, 1)
                config[key] = value.strip()
    for item in overrides:
        key, value = item.split("=", 1)
        config[key] = value
    return config


EXAMPLE_CONFIG = """\
# Example configuration of generate_multiparticle_gun.py (same format as the MLPF gun): a "particle soup" of
# 10 to 15 particles per event, in a cone ("jet") of radius 0.5 in (eta, phi) around an axis drawn for each event.
# Lines starting with # are ignored; the values can be changed on the command line with --set key=value.

# number of particles per event (uniform in this range)
npart_range 10,15

# directions: eta uniform in eta_range, or (to cover the endcaps better) cos(theta) uniform in costheta_range
eta_range -0.5,0.5
# costheta_range -0.98,0.98

# momentum [GeV], uniform in log(p)
mom_range 0.5,50

# radius of the cone around the axis of the event, in (eta, phi); with 0, each particle has its own direction
drmax 0.5

# PDG codes, each particle drawn uniformly from the list (repeat a code to give it more weight; with exactly two
# codes, they alternate)
pid_list 211,-211,130,2112,22,11,-11,13,-13,321,-321,2212,-2212

# number of events (overridden by -n)
nevents 100
"""


def floats(value):
    return [float(x) for x in value.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Multi-particle gun (HepMC3 output)")
    parser.add_argument("--config", default=None, help="configuration file (format of the MLPF gun)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="set or override a configuration value")
    parser.add_argument("-n", "--nEvents", type=int, default=None, help="number of events (default: nevents of the configuration)")
    parser.add_argument("-o", "--output", default="events.hepmc3", help="output HepMC3 ASCII file")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--write-example", nargs="?", const="soup.gun", default=None, metavar="FILE",
                        help="write an example configuration file (default name: soup.gun) and exit")
    args = parser.parse_args()

    if args.write_example:
        try:
            with open(args.write_example, "x") as f:
                f.write(EXAMPLE_CONFIG)
        except FileExistsError:
            raise SystemExit(f"{args.write_example} already exists: not overwritten")
        print(f"Example configuration written to {args.write_example}")
        return

    from pyHepMC3 import HepMC3 as hm  # only needed to generate the events

    cfg = read_config(args.config, args.set)
    nevents = args.nEvents if args.nEvents is not None else int(float(cfg["nevents"]))
    nmin, nmax = (int(x) for x in cfg["npart_range"].split(","))
    pmin, pmax = floats(cfg["mom_range"])
    drmax = float(cfg.get("drmax", 0))
    pids = [int(x) for x in cfg["pid_list"].split(",")]
    unknown = sorted({abs(p) for p in pids} - set(MASSES))
    if unknown:
        raise SystemExit(f"no mass for PDG codes {unknown}: add them to MASSES")
    if "costheta_range" in cfg:
        cmin, cmax = floats(cfg["costheta_range"])

        def draw_eta():
            return math.atanh(rng.uniform(cmin, cmax))  # eta = atanh(cos(theta))
    else:
        emin, emax = floats(cfg["eta_range"])

        def draw_eta():
            return rng.uniform(emin, emax)

    rng = random.Random(args.seed)
    writer = hm.WriterAscii(args.output)
    for ievt in range(nevents):
        event = hm.GenEvent(hm.Units.GEV, hm.Units.MM)
        event.set_event_number(ievt)
        vertex = hm.GenVertex(hm.FourVector(0.0, 0.0, 0.0, 0.0))
        for pz in (125.0, -125.0):
            beam = hm.GenParticle(hm.FourVector(0.0, 0.0, pz, 125.0), 11 if pz > 0 else -11, 4)
            vertex.add_particle_in(beam)
        axis_eta, axis_phi = draw_eta(), rng.uniform(-math.pi, math.pi)
        for i in range(rng.randint(nmin, nmax)):
            pid = pids[i % 2] if len(pids) == 2 else rng.choice(pids)
            if drmax == 0:
                eta, phi = draw_eta(), rng.uniform(-math.pi, math.pi)
            else:
                while True:
                    deta, dphi = rng.uniform(-drmax, drmax), rng.uniform(-drmax, drmax)
                    if math.hypot(deta, dphi) <= drmax:
                        break
                eta, phi = axis_eta + deta, axis_phi + dphi
            theta = 2.0 * math.atan(math.exp(-eta))
            p = math.exp(rng.uniform(math.log(pmin), math.log(pmax)))
            mass = MASSES[abs(pid)]
            momentum = hm.FourVector(p * math.sin(theta) * math.cos(phi), p * math.sin(theta) * math.sin(phi), p * math.cos(theta),
                                     math.sqrt(p * p + mass * mass))  # fmt: skip
            particle = hm.GenParticle(momentum, pid, 1)
            particle.set_generated_mass(mass)
            vertex.add_particle_out(particle)
        event.add_vertex(vertex)
        writer.write_event(event)
    writer.close()
    print(f"Wrote {nevents} events to {args.output}")


if __name__ == "__main__":
    main()
