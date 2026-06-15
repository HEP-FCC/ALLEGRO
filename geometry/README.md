# Print ALLEGRO parameters from compact files

1. setup the nightly
2. select subdetectors to show/skip in `printConstants.py`
3. execute script: `python printConstants.py`


# ALLEGRO ECal barrel calculations and sketches

The code can be run standalone or with the key4hep stack setup
In the former case, the user has to setup a virtual environment.
In the latter, the nightly has to be setup.
Note that the features of the script that read the geometry parameter from k4geo can only be executed in the latter case.
Jupytext is used to sync Jupyter notebooks with python scripts

## Initial setup

If using the script without the key4hep stack, the first time, setup a virtualenv environment with:

```
source setup.sh
```

## Execution

For the standalone version, load the environment with
```
source env.sh
```
Otherwise, setup the nightly

Then, start jupyter:
```
jupyter-lab
```

Execute notebook <code>Barrel geometry calculations.ipynb</code>.

If using the standalone version: when done, deactivate the virtualenv environment with
```
deactivate
```

## Workflow
  - Edit only .ipynb files in Jupyter
  - Commit both .ipynb and .py
  - Do not manually edit the .py corresponding to the .ipynb
  - Use jupytext --sync <file.ipynb> if files are out of sync
