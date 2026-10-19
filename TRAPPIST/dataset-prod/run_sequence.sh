#!/usr/bin/env bash

set -eEo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_sequence.sh [stages] [options]

Stages:
  --generate, -g       Generate and filter HepMC events
  --simulate, -s       Run detector simulation
  --reconstruct, -r    Run reconstruction
  --flatten, -f        Run parquet preprocessing (requires reconstruction)

Required options:
  --home DIR           TRAPPIST directory
  --output DIR         Final output directory
  --scratch DIR        Temporary working directory
  --seed N             Positive integer job seed
  --events N           Number of events (-1 means all input events)

Stage-specific options:
  --particle gamma|pi0 Required by --generate and --flatten
  --card FILE          Pythia card; required by --generate
                       A basename is looked up under dataset-prod/cards
  --keep-hepmc         Move the filtered generated HepMC file to OUTPUT/hepmc
  --input FILE         HepMC input for simulation, or EDM4hep simulation ROOT
                       input for reconstruction without simulation
  --skip-events N      Events to skip in an existing HepMC input (default: 0)

Examples:
  # Generate, simulate, reconstruct, and flatten
  run_sequence.sh -g -s -r -f \
    --home /path/to/TRAPPIST --output /path/to/output \
    --scratch /path/to/scratch --seed 1 --events 100 \
    --particle pi0 --card p8_ee_ZH-tautau_ecm240.cmd

  # Simulate an existing HepMC file, reconstruct, and flatten
  run_sequence.sh -s -r -f \
    --home /path/to/TRAPPIST --output /path/to/output \
    --scratch /path/to/scratch --seed 1 --events 100 \
    --particle pi0 --input /path/to/events.hepmc3

  # Reconstruct an existing simulation file
  run_sequence.sh -r \
    --home /path/to/TRAPPIST --output /path/to/output \
    --scratch /path/to/scratch --seed 1 --events 100 \
    --input /path/to/out_sim_edm4hep.root
EOF
}

die() {
    echo "Error: $*" >&2
    echo "Run '$0 --help' for usage." >&2
    exit 2
}

require_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

GEN=0
SIM=0
REC=0
FLATTEN=0
KEEP_HEPMC=0

HOMEDIR=""
OUTPUTDIR=""
SCRATCHDIR=""
SEED=""
EVENTS=""
PARTICLE=""
CARD=""
INPUTFILE=""
SKIP_EVENTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--generate)
            GEN=1
            shift
            ;;
        -s|--simulate)
            SIM=1
            shift
            ;;
        -r|--reconstruct)
            REC=1
            shift
            ;;
        -f|--flatten)
            FLATTEN=1
            shift
            ;;
        --keep-hepmc)
            KEEP_HEPMC=1
            shift
            ;;
        --home)
            require_value "$@"
            HOMEDIR=$2
            shift 2
            ;;
        --output)
            require_value "$@"
            OUTPUTDIR=$2
            shift 2
            ;;
        --scratch)
            require_value "$@"
            SCRATCHDIR=$2
            shift 2
            ;;
        --seed)
            require_value "$@"
            SEED=$2
            shift 2
            ;;
        --events)
            require_value "$@"
            EVENTS=$2
            shift 2
            ;;
        --particle)
            require_value "$@"
            PARTICLE=$2
            shift 2
            ;;
        --card)
            require_value "$@"
            CARD=$2
            shift 2
            ;;
        --input)
            require_value "$@"
            INPUTFILE=$2
            shift 2
            ;;
        --skip-events)
            require_value "$@"
            SKIP_EVENTS=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            [[ $# -eq 0 ]] || die "positional arguments are not supported"
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

(( GEN || SIM || REC || FLATTEN )) || die "select at least one stage"

[[ -n "$HOMEDIR" ]] || die "--home is required"
[[ -n "$OUTPUTDIR" ]] || die "--output is required"
[[ -n "$SCRATCHDIR" ]] || die "--scratch is required"
[[ -n "$SEED" ]] || die "--seed is required"
[[ -n "$EVENTS" ]] || die "--events is required"

[[ -d "$HOMEDIR" ]] || die "home directory does not exist: $HOMEDIR"
[[ "$SEED" =~ ^[1-9][0-9]*$ ]] || die "--seed must be a positive integer"
[[ "$EVENTS" == "-1" || "$EVENTS" =~ ^[1-9][0-9]*$ ]] || \
    die "--events must be a positive integer or -1"
[[ "$SKIP_EVENTS" =~ ^[0-9]+$ ]] || die "--skip-events must be a non-negative integer"

if [[ -n "$PARTICLE" && "$PARTICLE" != "gamma" && "$PARTICLE" != "pi0" ]]; then
    die "--particle must be 'gamma' or 'pi0'"
fi

if (( GEN || FLATTEN )); then
    [[ -n "$PARTICLE" ]] || die "--particle is required by generation and preprocessing"
fi

(( ! FLATTEN || REC )) || die "--flatten requires --reconstruct"
(( ! GEN || SIM || ! REC )) || \
    die "generation cannot feed reconstruction unless --simulate is also selected"

if (( GEN )); then
    [[ "$EVENTS" != "-1" ]] || die "generation requires a positive --events value"
    [[ -n "$CARD" ]] || die "--card is required by generation"
    [[ -z "$INPUTFILE" ]] || die "--input cannot be combined with --generate"
fi

(( ! KEEP_HEPMC || GEN )) || die "--keep-hepmc requires --generate"

if (( SIM )); then
    input_count=0
    [[ -n "$CARD" ]] && (( input_count += 1 ))
    [[ -n "$INPUTFILE" ]] && (( input_count += 1 ))
    (( input_count == 1 )) || \
        die "simulation requires exactly one of --card and --input"

    if (( GEN )); then
        [[ -n "$CARD" ]] || die "simulation after generation requires --card"
    else
        [[ -n "$INPUTFILE" ]] || die "simulation without generation requires --input"
        [[ -z "$CARD" ]] || die "--card requires --generate"
    fi

elif [[ -n "$INPUTFILE" ]]; then
    (( REC )) || die "--input requires --simulate or --reconstruct"
fi

if (( SKIP_EVENTS > 0 )) && (( ! SIM || GEN )); then
    die "--skip-events is only valid when simulating an existing --input file"
fi

if (( ! GEN && ! SIM )) && [[ -n "$CARD" ]]; then
    die "--card is only valid with --generate"
fi

if (( REC )); then
    if (( ! SIM )); then
        (( ! GEN )) || die "reconstruction after generation requires --simulate"
        [[ -n "$INPUTFILE" ]] || \
            die "reconstruction-only operation requires --input"
    fi
fi

CARD_SOURCE=""
if [[ -n "$CARD" ]]; then
    if [[ -f "$CARD" ]]; then
        CARD_SOURCE=$CARD
    else
        CARD_SOURCE="${HOMEDIR}/dataset-prod/cards/${CARD}"
    fi
    [[ -f "$CARD_SOURCE" ]] || die "card file does not exist: $CARD_SOURCE"
fi

if [[ -n "$INPUTFILE" ]]; then
    [[ -f "$INPUTFILE" ]] || die "input file does not exist: $INPUTFILE"
fi

JOBDIR="${SCRATCHDIR}/${SEED}"
PATH_ALLEGRO_DATA_RECO="${HOMEDIR}/dataset-prod/data"

show_value() {
    local value=$1
    [[ -n "$value" ]] && printf '%s' "$value" || printf '%s' '<not set>'
}

echo "Resolved configuration"
printf '  %-18s %s\n' "generate:" "$GEN"
printf '  %-18s %s\n' "simulate:" "$SIM"
printf '  %-18s %s\n' "reconstruct:" "$REC"
printf '  %-18s %s\n' "flatten:" "$FLATTEN"
printf '  %-18s %s\n' "keep HepMC:" "$KEEP_HEPMC"
printf '  %-18s %s\n' "home:" "$HOMEDIR"
printf '  %-18s %s\n' "output:" "$OUTPUTDIR"
printf '  %-18s %s\n' "scratch:" "$SCRATCHDIR"
printf '  %-18s %s\n' "job directory:" "$JOBDIR"
printf '  %-18s %s\n' "seed:" "$SEED"
printf '  %-18s %s\n' "events:" "$EVENTS"
printf '  %-18s %s\n' "particle:" "$(show_value "$PARTICLE")"
printf '  %-18s %s\n' "card:" "$(show_value "$CARD_SOURCE")"
printf '  %-18s %s\n' "input:" "$(show_value "$INPUTFILE")"
printf '  %-18s %s\n' "skip events:" "$SKIP_EVENTS"

# Change this line to select a different Key4hep release.
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

if (( REC )) && [[ ! -d "$PATH_ALLEGRO_DATA_RECO" ]]; then
    (
        cd "${HOMEDIR}/dataset-prod/utils"
        python downloadFilesForReco.py
    )
    echo "Downloaded reconstruction data into $PATH_ALLEGRO_DATA_RECO"
fi

mkdir -p "$OUTPUTDIR" "$JOBDIR"
cd "$JOBDIR"

export K4GEO="${HOMEDIR}/k4geo"
export FCCCONFIG="${HOMEDIR}/FCC-config"

echo "Working directory: $PWD"
echo "K4GEO: $K4GEO"
echo "FCCCONFIG: $FCCCONFIG"

if (( GEN )); then
    cp "$CARD_SOURCE" card.cmd
    echo "Running generation from $CARD_SOURCE"
    echo "Random:seed=${SEED}" >> card.cmd

    k4run "${HOMEDIR}/dataset-prod/utils/pythia.py" \
        -n "$EVENTS" \
        --Dumper.Filename "out_gen_${SEED}.hepmc" \
        --Pythia8.PythiaInterface.pythiacard card.cmd

    if [[ "$PARTICLE" == "gamma" ]]; then
        FILTER_SCRIPT="${HOMEDIR}/dataset-prod/utils/readHEPMC_photon.py"
    else
        FILTER_SCRIPT="${HOMEDIR}/dataset-prod/utils/readHEPMC_pi0.py"
    fi

    python "$FILTER_SCRIPT" \
        --input-file "out_gen_${SEED}.hepmc" \
        --output-file "out_filter_${SEED}.hepmc" \
        --evt_per_file "$EVENTS" \
        --seed "$SEED"
fi

if (( SIM )); then
    if (( GEN )); then
        SIM_INPUT="out_filter_${SEED}.hepmc"
        [[ -f "$SIM_INPUT" ]] || die "filtered generation output was not created: $SIM_INPUT"
    else
        SIM_INPUT=$INPUTFILE
    fi

    echo "Running simulation from $SIM_INPUT"
    ddsim \
        --compactFile "${K4GEO}/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml" \
        --inputFiles "$SIM_INPUT" \
        --outputFile "out_sim_edm4hep_${SEED}.root" \
        --numberOfEvents "$EVENTS" \
        --skipNEvents "$SKIP_EVENTS" \
        --random.enableEventSeed \
        --random.seed "$SEED"
fi

if (( GEN )); then
    # The unfiltered Pythia output is no longer needed after filtering.
    rm -f "out_gen_${SEED}.hepmc"

    FILTERED_HEPMC="out_filter_${SEED}.hepmc"
    if (( KEEP_HEPMC )); then
        HEPMC_OUTPUT_DIR="${OUTPUTDIR}/hepmc"
        HEPMC_OUTPUT_FILE="${HEPMC_OUTPUT_DIR}/${FILTERED_HEPMC}"
        mkdir -p "$HEPMC_OUTPUT_DIR"
        [[ ! -e "$HEPMC_OUTPUT_FILE" ]] || \
            die "refusing to overwrite existing HepMC output: $HEPMC_OUTPUT_FILE"
        mv "$FILTERED_HEPMC" "$HEPMC_OUTPUT_DIR/"
        echo "Stored filtered HepMC file: $HEPMC_OUTPUT_FILE"
    elif (( SIM )); then
        # Simulation has consumed the filtered file, so it can now be removed.
        rm -f "$FILTERED_HEPMC"
    fi
fi

if (( REC )); then
    if (( SIM )); then
        RECO_INPUT="out_sim_edm4hep_${SEED}.root"
    else
        RECO_INPUT=$INPUTFILE
    fi
    [[ -f "$RECO_INPUT" ]] || die "reconstruction input was not created: $RECO_INPUT"

    echo "Running reconstruction from $RECO_INPUT"
    ln -f -s "$PATH_ALLEGRO_DATA_RECO" .
    cp "${FCCCONFIG}/FCCee/FullSim/ALLEGRO/ALLEGRO_o1_v03/run_digi_reco.py" .

    k4run run_digi_reco.py \
        -n "$EVENTS" \
        --dataFolder data/ \
        --IOSvc.Input "$RECO_INPUT" \
        --IOSvc.Output "out_reco_edm4hep_${SEED}.root" \
        --doTopoClustering \
        --reconstructPi0s \
        --saveCells

    if (( SIM )) && [[ -f "out_reco_edm4hep_${SEED}.root" ]]; then
        rm -f "out_sim_edm4hep_${SEED}.root"
    fi
fi

if (( FLATTEN )); then
    RECO_OUTPUT="out_reco_edm4hep_${SEED}.root"
    [[ -f "$RECO_OUTPUT" ]] || die "reconstruction output was not created: $RECO_OUTPUT"

    echo "Running preprocessing on $RECO_OUTPUT"
    python "${HOMEDIR}/dataset-prod/utils/preprocessing.py" \
        --input-file "$RECO_OUTPUT" \
        --output-file "out_flatten_${SEED}" \
        --mode "$PARTICLE" \
        --events-per-file 100 > preprocessing_${SEED}.log 2>&1
    cat preprocessing_${SEED}.log

    shopt -s nullglob
    parquet_files=(*.parquet)
    if (( ${#parquet_files[@]} > 0 )); then
        cp "${parquet_files[@]}" "$OUTPUTDIR"
        touch "${OUTPUTDIR}/_SUCCESS_${SEED}"

        SUMMARY_FILE="${OUTPUTDIR}/_SUMMARY_${SEED}.txt"
        {
            echo "Parquet files summary for seed $SEED:"
            grep "Total events passing cuts:" preprocessing_${SEED}.log || echo "  (summary not found in preprocessing logs)"
        } | tee "$SUMMARY_FILE"
    fi
fi
