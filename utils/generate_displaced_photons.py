#!/usr/bin/env python
#
# Particle gun with a different vertex in every event, written as a HepMC3
# ASCII file for ddsim (--inputFiles <file>.hepmc3), e.g. for photon pointing
# studies. Two modes:
#
# --mode vertex (displaced vertices):
#  - vertex: z uniform in [-vertexZMax, vertexZMax], transverse radius
#    uniform in [vertexRMin, vertexRMax], azimuth uniform (mm), t = 0;
#  - direction: from the vertex to a point drawn uniformly on the front face
#    of the ECAL barrel (radius barrelRMin, |z| < targetZMax), so that the
#    barrel is covered uniformly whatever the vertex, and the incidence
#    angles follow from the vertex displacement.
#
# --mode surface (direction independent of the position, for training a
#   direction regression):
#  - start point on the cylinder r = startRadius in front of the ECAL
#    cryostat (default 2110 mm: between the tracker region, r < 2100 mm, and
#    the cryostat, r = 2149 mm), z uniform in [-startZMax, startZMax], phi
#    uniform;
#  - direction from two angles w.r.t. the radial direction at the start
#    point, drawn uniformly and independently of the position:
#    lambda (in the r-z plane, positive towards +z; for a photon from the
#    origin lambda = pi/2 - theta) in [-lambdaMax, lambdaMax], and alpha (in
#    the r-phi plane, positive towards +phi) in [-alphaMax, alphaMax]:
#    direction ~ e_r + tan(alpha) e_phi + tan(lambda) e_z.
#  The default startZMax (1900 mm) keeps the showers in the barrel (active
#  |z| < 3066 mm) for lambda up to 65 degrees.
#
# In both modes the energy is uniform in log(E) (default) or in E, in
# [energyMin, energyMax].
#
# Needs the Key4hep environment (pyHepMC3). Example:
#   python generate_displaced_photons.py -n 10000 -o photons_displaced.hepmc3 --seed 1
#   ddsim --inputFiles photons_displaced.hepmc3 --numberOfEvents 10000 \
#         --compactFile $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml \
#         --outputFile photons_displaced_sim.root
#
# The DD4hep HepMC3 reader starts a parent-less particle at the origin (or at
# its end vertex), and a particle with a parent at its production vertex. So
# each event has a dummy parent (same four-momentum, status 4 = beam, not
# simulated) ending at the displaced vertex, from which the particle
# (status 1) is produced. In MCParticles of the ddsim output the dummy parent
# comes first (generatorStatus 4): select the particle with generatorStatus 1.

import argparse
import math
import random

from pyHepMC3 import HepMC3 as hm


def main():
    parser = argparse.ArgumentParser(description='Photon gun with a per-event displaced vertex (HepMC3 output)')
    parser.add_argument('-n', '--nEvents', type=int, default=1000, help='number of events')
    parser.add_argument('-o', '--output', default='displaced_gun.hepmc3', help='output HepMC3 ASCII file')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--pdg', type=int, default=22, help='PDG code of the particle (massless assumed for 22)')
    parser.add_argument('--energyMin', type=float, default=0.5, help='minimum energy [GeV]')
    parser.add_argument('--energyMax', type=float, default=100., help='maximum energy [GeV]')
    parser.add_argument('--linearEnergy', action='store_true', help='energy uniform in E instead of log(E)')
    parser.add_argument('--vertexZMax', type=float, default=500., help='vertex z uniform in +-VERTEXZMAX [mm]')
    parser.add_argument('--vertexRMin', type=float, default=0., help='minimum vertex transverse radius [mm]')
    parser.add_argument('--vertexRMax', type=float, default=500., help='maximum vertex transverse radius [mm]')
    parser.add_argument('--barrelRMin', type=float, default=2172.8, help='radius of the ECAL barrel front face [mm]')
    parser.add_argument('--targetZMax', type=float, default=2500., help='impact points uniform in |z| < TARGETZMAX on the barrel front [mm]')
    parser.add_argument('--mode', choices=['vertex', 'surface'], default='vertex', help='displaced vertices, or start on a cylinder in front of the ECAL with uniform angles')
    parser.add_argument('--startRadius', type=float, default=2110., help='surface mode: radius of the start cylinder [mm]')
    parser.add_argument('--startZMax', type=float, default=1900., help='surface mode: start z uniform in +-STARTZMAX [mm]')
    parser.add_argument('--lambdaMax', type=float, default=65., help='surface mode: angle w.r.t. the radial direction in the r-z plane uniform in +-LAMBDAMAX [degrees]')
    parser.add_argument('--alphaMax', type=float, default=0.5, help='surface mode: angle w.r.t. the radial direction in the r-phi plane uniform in +-ALPHAMAX [rad]')
    args = parser.parse_args()

    if args.mode == 'vertex' and args.vertexRMax >= args.barrelRMin:
        parser.error('the vertex must be inside the barrel')
    if args.mode == 'surface' and not (args.lambdaMax < 90. and args.alphaMax < math.pi/2):
        parser.error('lambdaMax must be < 90 degrees and alphaMax < pi/2')
    mass = 0.
    if args.pdg != 22:
        # masses of the most common guns, GeV
        masses = {11: 0.000511, -11: 0.000511, 13: 0.10566, -13: 0.10566,
                  111: 0.13498, 211: 0.13957, -211: 0.13957}
        if args.pdg not in masses:
            parser.error(f'unknown mass for PDG {args.pdg}')
        mass = masses[args.pdg]

    rng = random.Random(args.seed)
    writer = hm.WriterAscii(args.output)
    for ievt in range(args.nEvents):
        if args.mode == 'vertex':
            # vertex
            zv = rng.uniform(-args.vertexZMax, args.vertexZMax)
            rv = rng.uniform(args.vertexRMin, args.vertexRMax)
            phiv = rng.uniform(-math.pi, math.pi)
            xv, yv = rv*math.cos(phiv), rv*math.sin(phiv)
            # impact point on the barrel front face and direction
            zi = rng.uniform(-args.targetZMax, args.targetZMax)
            phii = rng.uniform(-math.pi, math.pi)
            dx = args.barrelRMin*math.cos(phii) - xv
            dy = args.barrelRMin*math.sin(phii) - yv
            dz = zi - zv
        else:
            # start point on the cylinder and direction from the local angles
            zv = rng.uniform(-args.startZMax, args.startZMax)
            phiv = rng.uniform(-math.pi, math.pi)
            xv, yv = args.startRadius*math.cos(phiv), args.startRadius*math.sin(phiv)
            lam = math.radians(rng.uniform(-args.lambdaMax, args.lambdaMax))
            alpha = rng.uniform(-args.alphaMax, args.alphaMax)
            ta, tl = math.tan(alpha), math.tan(lam)
            dx = math.cos(phiv) - ta*math.sin(phiv)
            dy = math.sin(phiv) + ta*math.cos(phiv)
            dz = tl
        norm = math.sqrt(dx*dx + dy*dy + dz*dz)
        # energy
        if args.linearEnergy:
            energy = rng.uniform(args.energyMin, args.energyMax)
        else:
            energy = math.exp(rng.uniform(math.log(args.energyMin), math.log(args.energyMax)))
        p = math.sqrt(max(energy*energy - mass*mass, 0.))

        event = hm.GenEvent(hm.Units.GEV, hm.Units.MM)
        event.set_event_number(ievt)
        momentum = hm.FourVector(p*dx/norm, p*dy/norm, p*dz/norm, energy)
        parent = hm.GenParticle(momentum, args.pdg, 4)
        parent.set_generated_mass(mass)
        particle = hm.GenParticle(momentum, args.pdg, 1)
        particle.set_generated_mass(mass)
        vertex = hm.GenVertex(hm.FourVector(xv, yv, zv, 0.))
        vertex.add_particle_in(parent)
        vertex.add_particle_out(particle)
        event.add_vertex(vertex)
        writer.write_event(event)
    writer.close()
    print(f'Wrote {args.nEvents} events to {args.output}')


if __name__ == '__main__':
    main()
