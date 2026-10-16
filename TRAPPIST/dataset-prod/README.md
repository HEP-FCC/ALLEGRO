# Dataset production pipeline

This guide describes the dataset-production chain based on:

- [`submit.py`](submit.py), which creates and submits HTCondor jobs;
- [`run_sequence.sh`](run_sequence.sh), which runs one production job.

Run the commands from the TRAPPIST directory. In the examples below, replace
the output and scratch paths with your own locations. 
The settings used in this guide are the same adopted for the baseline study described in [this presentation](https://indico.cern.ch/event/1603697/#2-photon-pi0-classification-st).

## The chain at a glance

```text
Pythia card
    |
    v
generation -> HepMC filtering -> detector simulation -> reconstruction -> Parquet
                                  ^                     ^
                                  |                     |
                         existing HepMC file     existing simulation ROOT file
```

The four stages are:

| Stage | Option | What it does |
|---|---|---|
| Generation | `--generate` | Runs Pythia and selects an isolated gamma or pi0 event. |
| Simulation | `--simulate` | Runs DD4hep/Geant4 with the ALLEGRO detector. |
| Reconstruction | `--reconstruct` | Digitizes cells, builds topological clusters and reconstructs pi0 candidates. |
| Preprocessing | `--flatten` | Converts selected clusters and hits into Parquet files. |

Both scripts use the same stage options. Their single-letter aliases `-g`,
`-s`, `-r` and `-f` are also accepted.

## Before submitting jobs

`submit.py` creates a local `condor_std` directory for the HTCondor event
log, standard output and standard error files.

The scripts also expect these directories inside the TRAPPIST checkout:

```text
TRAPPIST/
├── dataset-prod/
├── FCC-config/
└── k4geo/
```

The runner loads the Key4hep nightly environment from CVMFS. The output
and scratch directories are created automatically.

### Quick smoke test

Before submitting a large campaign, test the full chain directly with
`run_sequence.sh` (see "Running one job directly" below) and a modest event
count, e.g. `--events 50`. Both the HepMC filter and the preprocessing cuts
are selective, so a handful of events can easily leave zero events at the
end of the chain: the job still exits successfully, but produces no Parquet
file and no `_SUCCESS` marker. This is expected statistical behavior, not a
failure. Requesting at least a few tens of events gives a more reliable
smoke test.

## Production with HTCondor

### Generate a new sample

This mode starts from a Pythia card and runs the complete chain:

```bash
python3 dataset-prod/submit.py \
    --homedir /path/to/TRAPPIST \
    --outdir /path/to/output \
    --tempdir /path/to/scratch \
    --card p8_ee_ZH-tautau_ecm240.cmd \
    --particle pi0 \
    --nev 1000 \
    --njobs 10 \
    --generate --simulate --reconstruct --flatten \
    --queue workday
```

This creates 10 independent generation jobs. Each job asks Pythia for 1,000
events and uses a different seed. Filtering is selective, so the number of
events reaching simulation can be smaller than 1,000.

Add `--keep-hepmc` to store each filtered generated file permanently under
`OUTDIR/hepmc`. The larger unfiltered Pythia file is deleted after filtering.
Without this flag, the filtered file is deleted after successful simulation.

Use `--particle gamma` to create the single-photon sample. The gamma filter
selects a pi0 from a tau decay and replaces it with a massless photon carrying
the same three-momentum.

### Process complete existing HepMC files

List all input files after one `--input` option. `--nev` is not allowed for
HepMC input (use the "Split existing files" mode below to process partial
ranges); omit `--njobs` as well:

```bash
python3 dataset-prod/submit.py \
    --homedir /path/to/TRAPPIST \
    --outdir /path/to/output \
    --tempdir /path/to/scratch \
    --input file1.hepmc3 file2.hepmc3 file3.hepmc3 \
    --particle pi0 \
    --simulate --reconstruct --flatten \
    --queue workday
```

One Condor job is created for each input file, and every job processes its file
completely:

```text
seed 1 -> file1.hepmc3 -> all events
seed 2 -> file2.hepmc3 -> all events
seed 3 -> file3.hepmc3 -> all events
```

Seeds start at 1 by default. Use `--seed` to choose the seed of the first job;
the following jobs receive consecutive values:

```bash
--seed 100
```

For the three files above, this assigns seeds 100, 101 and 102.

A shell wildcard can be used when the files are in one directory:

```bash
--input /path/to/generated/pi0/*.hepmc3
```

When retrying the same production (e.g. if a job is failed), run the same command again. Keep the same
input files in the same order, the same starting seed, and the same output and
scratch directories. Existing `_SUCCESS_SEED` markers will then skip jobs that
already finished successfully.

### Process already existing simulation ROOT files

Use the same `--input` structure, but enable reconstruction without
simulation. `--nev` is optional here and truncates each file to its first N
events (see below); omit `--njobs`, which is not supported for ROOT input:

```bash
python3 dataset-prod/submit.py \
    --homedir /path/to/TRAPPIST \
    --outdir /path/to/output \
    --tempdir /path/to/scratch \
    --input sim1.root sim2.root sim3.root \
    --particle pi0 \
    --reconstruct --flatten \
    --queue workday
```

One Condor job is created for each simulation file, and every file is
reconstructed completely:

```text
seed 1 -> sim1.root -> all events
seed 2 -> sim2.root -> all events
seed 3 -> sim3.root -> all events
```

With `--simulate`, both scripts interpret `--input` as HepMC. With
`--reconstruct` and no `--simulate`, they interpret `--input` as simulation
ROOT files. An optional `--nev` processes only the first requested events of
each ROOT file; ROOT inputs are not split with `--njobs`.

### Split existing files into smaller jobs

Specify `--nev` and `--njobs` only when the input files should be divided into
fixed-size ranges:

```bash
python3 dataset-prod/submit.py \
    --homedir /path/to/TRAPPIST \
    --outdir /path/to/output \
    --tempdir /path/to/scratch \
    --input file1.hepmc3 file2.hepmc3 \
    --particle pi0 \
    --nev 250 \
    --njobs 4 \
    --simulate --reconstruct --flatten \
    --queue workday
```

Here, `--njobs 4` means four jobs per input file. Each job reads 250 events:

```text
file1: offsets 0, 250, 500, 750
file2: offsets 0, 250, 500, 750
```

The input files must contain enough events for all requested ranges.

## Running one job directly

Use `run_sequence.sh` for tests or interactive production. Unlike the
submitter, it handles exactly one input and requires an explicit seed and event
count.

### Full chain from a Pythia card

```bash
dataset-prod/run_sequence.sh \
    --generate --simulate --reconstruct --flatten \
    --home /path/to/TRAPPIST \
    --output /path/to/output \
    --scratch /path/to/scratch \
    --seed 1 \
    --events 1000 \
    --particle pi0 \
    --card p8_ee_ZH-tautau_ecm240.cmd
```

The card can be an explicit path or a filename stored under
`dataset-prod/cards`.

### Full chain from an existing HepMC file

Use `--events -1` to process the complete file:

```bash
dataset-prod/run_sequence.sh \
    --simulate --reconstruct --flatten \
    --home /path/to/TRAPPIST \
    --output /path/to/output \
    --scratch /path/to/scratch \
    --seed 1 \
    --events -1 \
    --particle pi0 \
    --input /path/to/events.hepmc3
```

To process a range, give its size and offset:

```bash
--events 250 --skip-events 500
```

This processes 250 events after skipping the first 500.

### Reconstruction from an existing simulation file

This mode is available directly through the runner:

```bash
dataset-prod/run_sequence.sh \
    --reconstruct --flatten \
    --home /path/to/TRAPPIST \
    --output /path/to/output \
    --scratch /path/to/scratch \
    --seed 1 \
    --events -1 \
    --particle pi0 \
    --input /path/to/out_sim_edm4hep.root
```

Both `submit.py` and `run_sequence.sh` use `--input`. With simulation enabled,
it means a HepMC file. With reconstruction but no simulation, it means a
simulation ROOT file. The runner accepts one input file, while the submitter
accepts one or more.

## Valid stage combinations

The common combinations are:

| Options | Starting point | Result |
|---|---|---|
| `--generate` | Pythia card | Generated and filtered HepMC file in scratch. |
| `--generate --simulate` | Pythia card | Simulated ROOT file in scratch. |
| `--generate --simulate --reconstruct` | Pythia card | Reconstructed ROOT file in scratch. |
| `--generate --simulate --reconstruct --flatten` | Pythia card | Complete production ending in Parquet files. |
| `--simulate` | Existing HepMC | Simulated ROOT file in scratch. |
| `--simulate --reconstruct` | Existing HepMC | Reconstructed ROOT file in scratch. |
| `--simulate --reconstruct --flatten` | Existing HepMC | Complete production ending in Parquet files. |
| `--reconstruct` | Existing simulation ROOT file | Reconstructed ROOT file. |

Flattening requires reconstruction. Generation cannot feed reconstruction
without simulation. The particle mode is required for generation and
flattening.

## Files used by each stage

The entry scripts are in `dataset-prod`, but the chain also uses files from
other parts of the checkout:

| Purpose | File or directory |
|---|---|
| Pythia card | [`dataset-prod/cards/p8_ee_ZH-tautau_ecm240.cmd`](cards/p8_ee_ZH-tautau_ecm240.cmd) |
| Pythia steering | [`dataset-prod/utils/pythia.py`](utils/pythia.py), taken from [k4Gen](https://github.com/key4hep/k4Gen/blob/main/k4Gen/options/pythia.py) |
| pi0 filtering | [`dataset-prod/utils/readHEPMC_pi0.py`](utils/readHEPMC_pi0.py), selects an isolated pi0 from a tau decay within the barrel acceptance |
| Gamma filtering | [`dataset-prod/utils/readHEPMC_photon.py`](utils/readHEPMC_photon.py), same selection as above but replaces the pi0 with a same-momentum photon |
| ALLEGRO detector geometry | [`k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml`](../k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml) |
| Reconstruction steering | [`FCC-config/FCCee/FullSim/ALLEGRO/ALLEGRO_o1_v03/run_digi_reco.py`](../FCC-config/FCCee/FullSim/ALLEGRO/ALLEGRO_o1_v03/run_digi_reco.py), invoked with a fixed set of flags (`--doTopoClustering --reconstructPi0s --saveCells`) hardcoded in `run_sequence.sh`; it accepts more options than these (see its own `--help`) — to change them, edit the `k4run run_digi_reco.py` call in `run_sequence.sh` |
| Noise and neighbour maps | [`dataset-prod/data`](data/) |
| Reconstruction-data downloader | [`dataset-prod/utils/downloadFilesForReco.py`](utils/downloadFilesForReco.py), fetches each missing file once, with no retry or integrity check |
| Parquet preprocessing | [`dataset-prod/utils/preprocessing.py`](utils/preprocessing.py) |

If `dataset-prod/data` is missing when reconstruction starts, the runner calls
`downloadFilesForReco.py` to download the required maps and model files.

## Parquet preprocessing in detail

`preprocessing.py` reads the reconstructed EDM4hep file and, for each event,
picks at most one calorimeter cluster to keep, truth-matches it to the
generated pi0 or photon, and writes its hit-level features to Parquet.
`--mode pi0|gamma` is required and selects which truth particle and which
cuts below apply; `run_sequence.sh` always passes the same particle mode used
for generation/filtering.

Cluster selection is one of two mutually exclusive paths:

- **Default (leading unpaired cluster)**: the event is kept only if
  `AugmentedEMBCaloTopoClusters` contains at most 2 clusters, no pi0 has
  already been resolved by `Pi0RecoTool`, and there is exactly 1 unpaired
  cluster; that cluster is used. This is the rejection counted as
  "Cluster cut not passed" in the log.
- **`--bdt-cuts`**: instead tries to truth-match a resolved pi0 candidate
  first, falling back to the leading paired/unpaired cluster; rejections are
  counted as "BDT cuts not passed".

After a cluster is selected, truth/geometry cuts are applied (unless
`--geo-only` is given, which applies only the geometric part):

| Cut | What it requires | Constant |
|---|---|---|
| Geometric | Every generated photon's trajectory endpoint has `\|z\| < Z_MAX` and transverse radius `> R_MIN` (i.e. it reaches the calorimeter, not lost in the beampipe/support) | `R_MIN = 2100 mm`, `Z_MAX = 2350 mm` |
| Decay mode | Exactly 2 generated photons for pi0, exactly 1 for gamma (excludes Dalitz decays `pi0 -> e+ e- gamma`) | — |
| MC truth / energy sharing (pi0 only, skipped with `--geo-only`) | Using `CaloHitMCParticleLinks`, the most energetic truth photon contributing to the selected cluster's hits must carry `< FRACTION` of the cluster's energy, i.e. the cluster is not dominated by a single photon | `FRACTION = 0.95` |
| BDT-like truth matching (only with `--bdt-cuts`) | Angular distance `deltaR` between the truth photon/pi0 direction and the candidate cluster direction, and the cluster/MC energy ratio, must fall inside fixed windows | `BDT_DELTA_R_CUT = 0.15`, `BDT_LOWER_RATIO_E = 0.6`, `BDT_UPPER_RATIO_E = 1.5` |

These rejections are counted as "MC cut not passed" (geometric/MC-truth path)
or folded into "BDT cuts not passed" (BDT path); they are a separate, later
stage than generation-level HepMC filtering, so an event can pass the HepMC
filter but still be dropped here.

Events that survive are turned into one record each with:

- `X_hit`: per-hit `[energy, position.x, position.y, position.z]` for every
  hit of the selected cluster;
- `y_event_type`: `1` for pi0, `0` for gamma (same value repeated per hit);
- `z_mc_energy`, `z_pi0_angle`, `z_pi0_distance`: truth photon energy, and
  (pi0 mode only) the opening angle and endpoint distance between the two
  truth photons, zero-filled for gamma events.

Records are buffered and written every `--events-per-file` events (and once
more at the end for the remainder), producing `out_flatten_SEED_0000.parquet`,
`_0001.parquet`, etc. `run_sequence.sh` always passes `--events-per-file 100`
and no `--bdt-cuts`/`--geo-only`, so a production run uses the default
leading-unpaired-cluster path with the full geometric + MC-truth cuts.

## Outputs

Each job works under:

```text
SCRATCH/SEED/
```

During execution, and depending on the selected stages, this directory can
contain:

```text
out_gen_SEED.hepmc
out_filter_SEED.hepmc
out_sim_edm4hep_SEED.root
out_reco_edm4hep_SEED.root
out_flatten_SEED_*.parquet
```

For generation jobs, the unfiltered `out_gen_SEED.hepmc` file is deleted after
filtering. If simulation runs, the filtered file is also deleted after it has
been consumed unless `--keep-hepmc` is set. With that flag, it is moved to:

```text
OUTPUT/hepmc/out_filter_SEED.hepmc
```

In generation-only mode, the filtered file remains in scratch when the flag is
not set because it is the final product of that mode.

After successful reconstruction, a simulation file created by the same job is
deleted to save space. An external simulation file used for reconstruction-only
operation is not deleted.

After successful preprocessing, Parquet files are copied to `--outdir`, and
the runner creates:

```text
_SUCCESS_SEED
_SUMMARY_SEED.txt
```

`_SUMMARY_SEED.txt` reports the number of events available for that seed,
i.e. the "Total events passing cuts" line from the preprocessing log. The
full preprocessing log, with the counts rejected by each cut, is kept in
`SCRATCH/SEED/preprocessing_SEED.log`.

Each Parquet file stores all its events in a single row (one record whose
fields are per-event arrays), so the Parquet row count is always 1 and is not
the number of events. To count events in a file, use e.g.
`len(ak.from_parquet(f, columns="y_event_type")["y_event_type"])`.

On resubmission, `submit.py` skips seeds whose success marker already
exists. A seed where no event survives filtering or the preprocessing cuts
never gets a `_SUCCESS` marker, so it is resubmitted on every retry even
though the job itself did not fail; see the smoke-test note above.

## Quick help

```bash
dataset-prod/run_sequence.sh --help
python3 dataset-prod/submit.py --help
```

*This documentation was in part generated with AI and reviewed by a human.*
