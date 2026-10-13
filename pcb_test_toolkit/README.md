# PCB test toolkit

This toolkit separates waveform acquisition on the Windows instrument PC from cross-talk and capacity analysis on an offline computer. The original MATLAB scripts and historical data in the parent folder are unchanged.

## Choose a workflow

| Task | Computer | Start with |
| --- | --- | --- |
| Acquire cross-talk waveforms | Windows instrument PC | `run_acquire_xtalk.m` |
| Acquire one capacity waveform per channel | Windows instrument PC | `run_acquire_capacity.m` |
| Inspect one injection/readout pair on the scope | Windows instrument PC | `run_acquire_debug.m` |
| Analyze cross-talk tower lists | Offline MATLAB computer, including the Mac | `run_offline_xtalk.m` |
| Analyze capacity tower lists relative to T0 | Offline MATLAB computer, including the Mac | `run_offline_capacity.m` |

Each entry script adds the toolkit folder to the MATLAB path for the current session. It does not change the saved MATLAB path. Acquisition uses Instrument Control Toolbox and a working VISA connection. Offline cross-talk analysis uses Control System Toolbox; its study plots also use Signal Processing Toolbox. Capacity waveform fitting uses Optimization Toolbox.

On the acquisition oscilloscope, CH1 carries the direct generator reference and stays at the settings chosen on the scope. CH2 carries the multiplexer readout. The Windows scripts adjust and read CH2 only. Set the scope's trigger source and CH1 display on the instrument before acquisition.

To call a `pcbtest` function directly from the MATLAB Command Window, first add the toolkit folder with `addpath('path/to/pcb_test_toolkit')`. The entry scripts do this automatically.

## Script-by-script guide

### `run_acquire_xtalk.m` — acquire cross-talk

**Purpose:** Acquire one physically connected tower per run. The multiplexer switches injection and readout channels within that tower to capture the complete 15 x 15 matrix (225 waveforms). It cannot switch towers.

**Before running:**

1. In MATLAB, run `visadevlist` and check the example `matrixResource` and `scopeResource` values against the connected instruments. These values came from the tested T4 setup and may differ on another PC. The list's row numbers can change, so use resource names rather than row numbers.
2. The example uses `tower = 4`, `datasetLabel = 'PCBCHINA'`, and `dataRoot = fullfile(toolkitDir, 'data', 'xtalk_run_003')`. Set `tower` to the tower currently connected and choose a new `dataRoot` for another campaign.
3. Set the CH2 `scopeSignalRange`, `scopeAdjacentRange`, and `scopeOtherRange` values at the top of the script. The matching `scopeSignalPosition`, `scopeAdjacentPosition`, `scopeOtherPosition`, and three `scope...Offset` values are editable there too. The script selects one complete setting for every injection/readout pair. To keep one range for all pairs, set all three range values to the same number. The tested T4 example waits `readDelaySeconds = 0.5` seconds between completed acquisition and waveform readout. Check the channel routing in `+pcbtest/set_matrix_channel.m` against the actual multiplexer wiring.
4. Optionally set `matrixIdContains` and `scopeIdContains` to text expected in the devices' `*IDN?` responses. Set `skipExistingFiles = true` to skip existing files with matching metadata; set it to `false` to reacquire and overwrite them.
5. For a short trial, change `cfg.channels` to `1:2` and use a separate trial `dataRoot`. For a full run, use `1:15`.
6. Physically connect the selected tower and run `run_acquire_xtalk.m`. Confirm the connection at the prompt to start its 225 waveforms, or type `q` to stop.
7. To measure another tower, change the physical connection, update `tower`, and run the script again.

**Output:** `dataRoot/T4/T4_xtalk_PCBCHINA_01-02.mat` is an example. Each file contains `t`, CH2 `data`, and `metadata`; a full tower produces 225 files. With `skipExistingFiles = true`, existing pair files are skipped only when metadata matches the current pair, tower, readout scope channel, scope settings, and instrument identities. A mismatch stops acquisition with an error. The tested T4 example has `skipExistingFiles = false`, so rerunning it reacquires and overwrites existing pair files. After changing any CH2 setting for an existing dataset, use a new `dataRoot` or keep that switch false to reacquire its files.

### `run_acquire_capacity.m` — acquire capacity waveforms

**Purpose:** Acquire one physically connected tower per run. The multiplexer routes injection and readout to each of its 15 channels in turn, producing one rising waveform per channel. Tau fitting happens offline.

**Before running:**

1. Check the example VISA resource names against `visadevlist`. Set `tower = 0` and connect T0 for the reference acquisition first.
2. The example uses `dataRoot = fullfile(toolkitDir, 'data', 'capacity_run_001')` and `datasetLabel = 'PCBCHINA'`. Set `scopeRange` at the top of the script to the desired CH2 `CHANnel:RANGe` command value for all 15 channels. Position, offset, and pre-acquisition settle time come from `cfg.capacity.scope` and `cfg.capacity.settleSeconds` in `+pcbtest/default_config.m`. `readDelaySeconds` adds a 1-second wait after completed acquisition before CH2 readout. Set `skipExistingFiles = true` to skip existing files with matching metadata; set it to `false` to reacquire and overwrite them.
3. Run `run_acquire_capacity.m` and confirm the T0 connection at the prompt. The script acquires its 15 channels once.
4. To measure another tower, change the physical connection, set `tower` to its number, and run the script again. Keep the same `dataRoot` and `datasetLabel` so offline analysis can find both that tower and T0.

**Output:** `dataRoot/T4/T4_Capacity_PCBCHINA_1.mat` is an example. Each file contains `t`, CH2 `data_read`, and `metadata`; a full tower produces 15 files. With `skipExistingFiles = true`, existing channel files are skipped only when metadata matches; a mismatch stops acquisition with an error. After changing the range for an existing dataset, use a new `dataRoot` or set `skipExistingFiles = false` to reacquire and overwrite its files.

### `run_acquire_debug.m` — inspect one waveform

**Purpose:** Route one chosen injection/readout pair and trigger exactly one scope acquisition for inspection. Saving is controlled by `saveWaveform`, which defaults to `false`.

**Before running:** The example uses T4, injection/readout channel 1, and the VISA resource names from the tested setup. Check `tower`, `injectChannel`, `readoutChannel`, `matrixResource`, and `scopeResource` before each run. `applyScopeSettings = true` by default, so the script sends `scopePosition`, `scopeRange`, and `scopeOffset` to CH2 once before acquisition. Set it to `false` to keep the CH2 vertical settings adjusted on the scope front panel; in that mode, the three numeric script values are ignored. CH1 remains the direct reference. `scopeRange` is the direct `CHANnel:RANGe` command value; consult the scope's command reference when translating a front-panel V/div value. Adjust `readoutScopeChannel` and `settleSeconds` if needed; `readDelaySeconds` defaults to 1 second after acquisition.

**How to use:** Connect the selected tower, set the desired channels in `run_acquire_debug.m`, and run `run_acquire_debug` in MATLAB. Acquisition starts immediately, with no confirmation prompt. Inspect the captured waveform on the scope. Set `saveWaveform = true` before running if you want a MAT file; leave it `false` to inspect without saving. Edit the channels and run the script again for another pair or after changing the scope settings. To compare with the xtalk acquisition, use the same injection/readout pair and match CH2 position, range, and offset; xtalk selects different settings for diagonal, adjacent, and other pairs. It does not perform a channel loop.

**Output:** Saved waveforms go to `output/debug/T<number>` by default with a timestamped name such as `T13_debug_03-09_20261008_153000_123.mat`. Each file contains `t`, CH2 `data`, and `metadata`. The debug files use their own filenames and are not selected by the offline matrix or capacity analyzers.

### `run_offline_xtalk.m` — analyze cross-talk

**Purpose:** Loop over cross-talk towers and run `pcbtest.analyze_xtalk` for each matrix. The example settings expect T4 data under `data/xtalk_run_001/T4`.

**Before running:** Set `towers`, `dataDir`, `datasetLabel`, `outputLabel`, `shapingTimes` (seconds), and `studyCells` at the top of the script. Copy the Windows data folder to the offline computer or point `dataDir` to its location. `outputLabel` names a folder under `pcb_test_toolkit/output`; set it to a new name such as `'run_2026_10_02'` to keep another analysis separate. `studyCells` lists injection cell positions; the initial value is `[3, 8]`, and `[]` skips the two waveform study views. For Windows toolkit data, set `dataDir` to the acquisition `dataRoot` and keep `towerSubfolders = true`. For historical data with all towers in one folder, set `towerSubfolders = false`. Then run `run_offline_xtalk.m`.

**Output for each tower and shaping time:**

- `results_T13_20ns.mat`: matrices, normalization denominator, display-channel mapping, and summary values.
- `pairs_T13_20ns.csv`: one row per injected/readout channel pair, with raw amplitudes and the globally normalized value.
- `T13_xtalk_20ns_plot.pdf`: the legacy-style cross-talk heatmap. Each pair uses its own shaped waveform's maximum magnitude, divided by the largest maximum in the complete 15 x 15 matrix. It uses the original turbo colors and fixed color range.
- `T13_peak_xtalk_20ns_plot.pdf`: the signal-peak amplitude map in volts. Each pair is sampled at the peak time of its injected channel's diagonal waveform, rather than using the pair's own maximum.
- `plots_ch/T13_xtalk_curves_20ns_ch3_plot.pdf`: all readout waveforms for one injection cell, shaped, high-pass filtered, normalized, and offset in time as in the first historical study figure.
- `plots_ch/T13_xtalk_curves_20ns_ch3_overlay_plot.pdf`: the second historical study figure, with all shaped readout waveforms overlaid without offsets or the extra high-pass filter. The channel number in each filename follows the cell-to-channel map.

Results go to `pcb_test_toolkit/output/<outputLabel>/xtalk/T<number>`. The toolkit's `.gitignore` excludes the entire `output/` folder. If an input file is missing, malformed, or has a different time axis, the analyzer stops and names the affected file.

### `run_offline_capacity.m` — analyze capacity

**Purpose:** Loop over capacity towers, fit their raw waveforms and the T0 reference, then calculate per-channel capacitance. The T0 tau array is fitted once and reused. The example settings expect `data/capacity_run_001/T0` and `data/capacity_run_001/T4`.

**Before running:** Set `towers` (exclude T0, which is read as a reference), `dataDir`, `datasetLabel`, and `outputLabel`. Copy both T0 and measurement tower data to the offline computer or point `dataDir` to their common parent folder. Use the same `outputLabel` as the xtalk script to collect both analyses under one folder, or choose different names for separate runs. For Windows toolkit data, keep `towerSubfolders = true`. For historical data with all towers in one folder, set it to `false`. Then run `run_offline_capacity.m`.

**Output for each tower:** `capacity_T13.mat` contains the fitted tau arrays, T0 tau array, capacitance in pF, component-fit coefficients, and display mapping; `capacity_T13.csv` gives one row per acquisition channel; `capacity_T13.pdf` shows the legacy-style capacity stairs plot in acquisition-channel order. `T13_capacity_fit.pdf` and `T13_capacity_fit.csv` show the line and coupling components in mapped cell order. Results go to `pcb_test_toolkit/output/<outputLabel>/capacity/T<number>`.

### `+pcbtest/default_config.m` — shared settings

**Purpose:** Return a configuration struct for the chosen tower. It contains the 15-channel display order, cross-talk shaping and baseline defaults, cross-talk and capacity scope settings, capacity fitting settings, and hardware connection fields. Tower 0 is supported as the capacity reference.

**Scope scaling defaults:** For cross-talk, diagonal (signal) pairs use CH2 position `-4`, range `5`, and offset `-1.8`; adjacent channel numbers use CH2 position `0`, range `0.15`, and offset `0.009`; other pairs use CH2 position `0`, range `0.08`, and offset `0.009`. Capacity uses CH2 position `-4`, range `2`, and offset `0`. The xtalk entry script exposes all three settings for each pair category near the top and overrides these defaults for that run. The acquisition functions send the values directly as `CHANnel2:POSition`, `CHANnel2:RANGe`, and `CHANnel2:OFFset` SCPI commands; they do not calculate an automatic scale. CH1 is not configured or read by the scripts. After `SING` completes, the scripts wait `readDelaySeconds = 1` second before reading CH2. Check the oscilloscope's command reference before converting a front-panel V/div setting into a `RANGe` value.

**How to use:** Call `cfg = pcbtest.default_config(13);` from an entry script, then override settings for that specific run. Supported tower groups and their display orders are defined in this file. For a tower outside these groups, add its channel mapping in this function before using it.

### `+pcbtest/acquire_xtalk.m` — instrument control and waveform saving

**Purpose:** Connect to the configured VISA resources, identify the instruments, route each injected/readout pair through the multiplexer, set the scope range, trigger a single acquisition, and save the waveform.

**How to use:** Normally called by `run_acquire_xtalk.m` once per tower. For a custom driver script, create a configuration with `pcbtest.default_config`, fill in the hardware resources and output directory, then call `pcbtest.acquire_xtalk(cfg)`. The multiplexer commands are in `+pcbtest/set_matrix_channel.m`.

### `+pcbtest/acquire_capacity.m` — capture capacity waveforms

**Purpose:** For each of 15 channels, route injection and readout to that channel, configure the scope, trigger one waveform, and save `t`, `data_read`, and metadata. The fitted tau is calculated later on the offline computer.

**How to use:** Normally called by `run_acquire_capacity.m` once for the connected tower. Acquire T0 in a separate run before the measurement towers. In a custom script, set `cfg.filePrefix` to a name such as `T13_Capacity_PCBCERN_`, then call `pcbtest.acquire_capacity(cfg)`.

### `+pcbtest/acquire_debug.m` — single acquisition and optional save

**Purpose:** Route the selected injection/readout pair, optionally apply scope settings, and capture one waveform. Save it when `cfg.debug.saveWaveform` is true.

**How to use:** Called by `run_acquire_debug.m`. Configure the entry script and run it; the helper manages the instrument connections and stores a timestamped MAT file when `saveWaveform = true`.

### `+pcbtest/analyze_xtalk.m` — matrix calculation and exports

**Purpose:** Read all 225 pair files, subtract each waveform's baseline, apply shaping, calculate cross-talk amplitudes, reorder channels for display, and write MAT, CSV, and PDF results.

**How to use:** Normally called by `run_offline_xtalk.m`. From another script, call `results = pcbtest.analyze_xtalk(cfg);` after setting `cfg.dataDir`, `cfg.filePrefix`, `cfg.outputDir`, and `cfg.shapingTimes`. The returned `results` struct array has one element per shaping time.

### `+pcbtest/plot_xtalk_study.m` — inspect all shaped cross-talk traces

**Purpose:** Recreate the two figures in `GPIB_ethernet_Orsay_proto3_check_xtalk_study.m` for each selected injection cell and shaping time. The offset view uses the historical 500 Hz second-order high-pass filter after shaping; the overlay view uses shaping alone. Both views are saved as vector PDFs.

**How to use:** `run_offline_xtalk.m` calls this function after the matrix analysis. Set `cfg.studyCells` to cell positions 1 through 15, or to `[]` to skip these plots. The historical study mapping is stored separately as `cfg.studyCellToChannel` in `default_config.m` because its cell order differs from the heatmap order. For a direct call, use `pcbtest.plot_xtalk_study(cfg);` after filling in the same input and output fields used by `analyze_xtalk`.

### `+pcbtest/analyze_capacity.m` — fit tau and calculate capacity

**Purpose:** Read 15 capacity waveforms for one tower and 15 T0 reference waveforms, fit the rise time constant of each, calculate `C = (tau_tower - tau_T0)/R`, and export MAT, CSV, and PDF results. It can alternatively read existing `tau_array` MAT files.

**How to use:** Normally called by `run_offline_capacity.m`. For a custom script, set the tower's `cfg.dataDir` and `cfg.filePrefix`, `cfg.capacity.referenceDir` and `cfg.capacity.referencePrefix` for T0, and `cfg.outputDir`; then call `result = pcbtest.analyze_capacity(cfg);`. The default input is raw waveforms. To use saved tau arrays instead, set `cfg.capacity.source = 'tauFiles'`, `cfg.capacity.tauDir`, and `cfg.capacity.tauPattern` (a `sprintf` pattern containing `%d` for the tower number). The legacy line-length and coupling-component fit is enabled by default using geometry values in `default_config.m`.

### `+pcbtest/fit_tau.m` — fit one capacity waveform

**Purpose:** Subtract the waveform's first sample, find the first crossing of the configured threshold (0.04 V by default), and fit `A*(1-exp(-(t-t0)/tau))` to the remaining samples using `lsqcurvefit`.

**How to use:** Called by `analyze_capacity`. To inspect one channel directly, load it with `pcbtest.load_waveform`, then call `[tau, amplitude, t0] = pcbtest.fit_tau(t, data, cfg.capacity);`.

### `+pcbtest/set_matrix_channel.m` — route one matrix channel

**Purpose:** Send the historical matrix commands for injection or readout on channels 1 through 15.

**How to use:** Called by both acquisition functions. Check this file when the multiplexer wiring or command map changes.

### `+pcbtest/load_waveform.m` — read one waveform

**Purpose:** Load a `.mat` file and check that its time and data vectors are finite, equally sized, and uniformly sampled. It reads historical cross-talk files with `t` and `data`, as well as files with `t` and `data_read`.

**How to use:** Called by the analyzer. To inspect one file directly, use `[t, data, metadata] = pcbtest.load_waveform(filePath);`. `metadata` is an empty struct for historical files without that field.

### `+pcbtest/shape_waveform.m` — apply the shaping filter

**Purpose:** Apply the third-order shaping response used by the historical analysis. Filter coefficients are cached when successive waveforms use the same sample interval and shaping time.

**How to use:** Called by the analyzer after baseline subtraction. For a single waveform, call `shaped = pcbtest.shape_waveform(data, t(2)-t(1), 20e-9);`.

### `+pcbtest/read_scope.m` — read one scope channel

**Purpose:** Read the selected scope channel's waveform header and ASCII sample values through the configured VISA connection, and return `t` in seconds and `data` in volts. The Windows acquisition scripts pass CH2 as the selected channel.

**How to use:** Called by both acquisition functions after a single acquisition. The scope command strings are defined here if the oscilloscope model changes.

## Waveform file format

Cross-talk files are named `<filePrefix><inject:02d>-<readout:02d>.mat` and contain `t` (seconds) and `data` (volts). Capacity files are named `<filePrefix><channel>.mat` and contain `t` and `data_read`. New acquisitions also include a `metadata` struct with acquisition time, tower, channel, scope settings, and instrument identities.

## Cross-talk calculation

For each waveform, subtract the mean of its first 10% of samples. Apply `H(s) = (tau*s)/(tau*s + 1)^3`, discretized with Tustin's method at the waveform sample interval. `rawMaxV(i,j)` is the maximum absolute shaped amplitude. `rawAtInjectionPeakV(i,j)` is the absolute shaped amplitude at the peak time of injected channel `i`'s diagonal waveform. `relativeGlobal` divides `rawMaxV` by the largest value anywhere in the 15 x 15 matrix. No injection-background subtraction is applied.

The heatmap reorders channels using `cfg.displayOrder`, sets the diagonal to zero, and omits `cfg.displayOmitPosition` (8 by default). The MAT and CSV files keep all 15 acquisition channels. The heatmap axes show display positions 1:14; `displayChannels` in the MAT file maps those positions back to acquisition channels.

For the historical injection view, the original script reloads the same cross-talk matrix when injection-background subtraction is disabled. The toolkit preserves that plot and filename, so its values match the main heatmap until a separate injection reference is introduced.

## Capacity calculation

The default capacity analysis fits each raw waveform with a rising exponential and uses T0 as the reference. For each acquisition channel, `capacitancePf = (tauSeconds - referenceTauSeconds) / resistanceOhm * 1e12`; resistance defaults to 9150 ohms. The CSV and MAT files retain all 15 channels. Both PDFs omit position 8; the plain plot uses acquisition order, and the fit plot applies the tower's display order. The result is capacitance in pF, without division by strip length.

The component fit uses the legacy strip lengths and coupling counts in `default_config.m` to estimate a line contribution in pF/mm and a coupling contribution in pF/count. It writes `T13_capacity_fit.csv` and `T13_capacity_fit.pdf` by default. Set `cfg.capacity.fitComponents = false` if a PCB uses different geometry or the component fit is not needed. The plain PDF keeps acquisition-channel order from the historical script; the component-fit PDF uses the tower's mapped cell order.
