# Print ALLEGRO parameters from compact files

1. setup the nightly
2. select subdetectors to show/skip in `printConstants.py`
3. execute script: `python printConstants.py`


# ALLEGRO ECal barrel calculations

Do not setup the nightly (use a fresh shell if needed)

## Initial setup

The first time only do:

```
source setup.sh
```

## Execution
```
source env.sh
jupyter-notebook
```
(or `jupyter-lab`, as you prefer)

Execute notebook <code>Barrel geometry calculations.ipynb</code>.
The font that is used by default might be missing in your system, you can set it with <code>p["font.sans-serif"] = ["font name"]</code> 
