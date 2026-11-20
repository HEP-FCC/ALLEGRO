#!/usr/bin/env python3

import os
import glob
import argparse


# ________________________________________________
def main():

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Submit dataset production jobs. With --simulate, --input files are "
            "HepMC; with --reconstruct and no --simulate, they are simulation "
            "ROOT files."
        )
    )

    parser.add_argument("--homedir", help="home directory from where script is run", required=True)
    parser.add_argument("--outdir", help="output directory")
    parser.add_argument("--tempdir", help="temporary directory")
    parser.add_argument(
        "--script",
        help="bash pipeline script",
        default="dataset-prod/run_sequence.sh",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help=(
            "HepMC files with --simulate, or simulation ROOT files with "
            "--reconstruct and no --simulate; omit --nev to process each file "
            "completely"
        ),
        default=[],
    )
    parser.add_argument(
        "--njobs",
        help="generation jobs, or HepMC chunks per input when --nev is used",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--seed",
        help="seed assigned to the first job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--nev",
        help="events per job; omit to process complete input files",
        default=None,
        type=int,
    )
    parser.add_argument("--card", help="pythia card", default="")
    parser.add_argument("--particle", choices=["gamma", "pi0"])
    parser.add_argument("--generate", "-g", dest="gen", action="store_true")
    parser.add_argument("--simulate", "-s", dest="sim", action="store_true")
    parser.add_argument("--reconstruct", "-r", dest="rec", action="store_true")
    parser.add_argument("--flatten", "-f", action="store_true")
    parser.add_argument(
        "--keep-hepmc",
        action="store_true",
        help="store filtered generated HepMC files under OUTDIR/hepmc",
    )
    parser.add_argument(
        "--queue",
        choices=[
            "espresso",
            "microcentury",
            "longlunch",
            "workday",
            "tomorrow",
            "testmatch",
            "nextweek",
        ],
        default="workday",
    )

    args = parser.parse_args()

    # ________________________________________________

    homedir = os.path.abspath(args.homedir)
    outdir = os.path.abspath(args.outdir)
    tempdir = os.path.abspath(args.tempdir)
    inputfiles = args.input
    script = args.script
    njobs = args.njobs
    first_seed = args.seed
    nev = args.nev
    card = args.card
    particle = args.particle
    queue = args.queue

    gen = args.gen
    sim = args.sim
    rec = args.rec
    flatten = args.flatten
    keep_hepmc = args.keep_hepmc

    if first_seed < 1:
        parser.error("--seed must be greater than zero")
    if keep_hepmc and not gen:
        parser.error("--keep-hepmc requires --generate")

    if gen:
        if inputfiles:
            parser.error("--input cannot be combined with --generate")
        generation_jobs = 2 if njobs is None else njobs
        generation_events = 100 if nev is None else nev
        if generation_jobs < 1 or generation_events < 1:
            parser.error("generation requires positive --njobs and --nev values")
        job_specs = [
            (None, 0, generation_events)
            for _ in range(generation_jobs)
        ]
    elif sim:
        if not inputfiles:
            parser.error("--simulate without --generate requires at least one --input")

        # By default, create one job per input and process every event.
        if nev is None or nev == -1:
            if njobs not in (None, 1):
                parser.error("omit --njobs when processing complete input files")
            job_specs = [
                (inputfile, 0, -1)
                for inputfile in inputfiles
            ]
        else:
            jobs_per_input = 1 if njobs is None else njobs
            if nev < 1 or jobs_per_input < 1:
                parser.error("chunked input processing requires positive --nev and --njobs values")
            job_specs = [
                (inputfile, chunk * nev, nev)
                for inputfile in inputfiles
                for chunk in range(jobs_per_input)
            ]
    elif rec:
        if not inputfiles:
            parser.error(
                "--reconstruct without --simulate requires at least one simulation ROOT input"
            )
        if njobs not in (None, 1):
            parser.error("--njobs is not supported for simulation ROOT inputs")

        reconstruction_events = -1 if nev is None or nev == -1 else nev
        if reconstruction_events < 1 and reconstruction_events != -1:
            parser.error("--nev must be greater than zero or -1")
        job_specs = [
            (inputfile, 0, reconstruction_events)
            for inputfile in inputfiles
        ]
    else:
        if inputfiles:
            parser.error("--input requires --simulate or --reconstruct")
        # Preserve the old defaults for workflows that do not run simulation.
        other_jobs = 2 if njobs is None else njobs
        other_events = 100 if nev is None else nev
        job_specs = [
            (None, 0, other_events)
            for _ in range(other_jobs)
        ]

    def is_done(seed):
        return os.path.exists(os.path.join(outdir, f"_SUCCESS_{seed}"))

    os.makedirs("condor_std", exist_ok=True)

    # condor submission header
    cmdfile = f"""
executable = {script}

output = condor_std/condor.$(ClusterId).$(ProcId).out
error  = condor_std/condor.$(ClusterId).$(ProcId).err
log    = condor_std/condor.$(ClusterId).log

+AccountingGroup = "group_u_FCC.local_gen"
should_transfer_files = YES

+JobFlavour = "{queue}"

"""
    jobCount = 0

    for job, (job_input, skip_events, job_events) in enumerate(job_specs):
        seed = first_seed + job
        # skip fully completed jobs
        if is_done(seed):
            print(f"seed {seed} already completed")
            continue

        # Build the arguments expected by run_sequence.sh.
        arguments = []
        if gen:
            arguments.append("--generate")
        if sim:
            arguments.append("--simulate")
        if rec:
            arguments.append("--reconstruct")
        if flatten:
            arguments.append("--flatten")
        if keep_hepmc:
            arguments.append("--keep-hepmc")

        arguments.extend(
            [
                "--home", homedir,
                "--output", outdir,
                "--scratch", tempdir,
                "--seed", str(seed),
                "--events", str(job_events),
            ]
        )

        if particle is not None:
            arguments.extend(["--particle", particle])

        if gen:
            arguments.extend(["--card", card])
        elif sim:
            arguments.extend(
                [
                    "--input", job_input,
                    "--skip-events", str(skip_events),
                ]
            )
        elif rec:
            arguments.extend(["--input", job_input])

        arguments = " ".join(arguments)
        cmdfile += f'arguments = "{arguments}"\n'
        cmdfile += "queue\n"

        jobCount += 1

        # print first command for debugging
        if jobCount == 1:
            debug_cmd = f"{script} {arguments}"
            print("\nExample command:")
            print(debug_cmd)
            print("")

    # write condor submission file
    with open("condor_ALLEGRO.sub", "w") as f:
        f.write(cmdfile)

    # submit jobs
    if jobCount > 0:
        print(f"\nSubmitting {jobCount} jobs")
        os.system("condor_submit condor_ALLEGRO.sub")
    else:
        print("\nNo jobs to submit.")


# ________________________________________________
if __name__ == "__main__":
    main()
