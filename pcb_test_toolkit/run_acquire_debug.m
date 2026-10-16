% Capture one selected injection/readout pair for scope setup.
toolkitDir = fileparts(mfilename('fullpath'));
addpath(toolkitDir);

% Edit these settings for the next single-shot acquisition.
tower = 4;
injectChannel = 1;
readoutChannel = 1;
matrixResource = 'GPIB0::7::INSTR';
scopeResource = 'USB0::0x0AAD::0x01D6::105097::0::INSTR';
matrixIdContains = '';         % Optional expected text from *IDN?.
scopeIdContains = '';          % Optional expected text from *IDN?.
readoutScopeChannel = 2;       % CH1 is the untouched generator reference.
readDelaySeconds = 1;         % Wait after the single acquisition before reading CH2.
settleSeconds = 0;

% Set false to keep the vertical settings adjusted on the scope itself.
applyScopeSettings = true;
scopePosition = 0;
scopeRange = 5;                % Sent to CHANnel:RANGe when applyScopeSettings is true.
scopeOffset = 0.009;

saveWaveform = false;          % Set true to save this single waveform.
saveRoot = fullfile(toolkitDir, 'output', 'debug');

cfg = pcbtest.default_config(tower);
cfg.hardware.matrixResource = matrixResource;
cfg.hardware.scopeResource = scopeResource;
cfg.hardware.matrixIdContains = matrixIdContains;
cfg.hardware.scopeIdContains = scopeIdContains;
cfg.scope.readoutChannel = readoutScopeChannel;
cfg.scope.readDelaySeconds = readDelaySeconds;
cfg.debug.injectChannel = injectChannel;
cfg.debug.readoutChannel = readoutChannel;
cfg.debug.applyScopeSettings = applyScopeSettings;
cfg.debug.scopeSettings = struct('position', scopePosition, ...
    'range', scopeRange, 'offset', scopeOffset);
cfg.debug.settleSeconds = settleSeconds;
cfg.debug.saveWaveform = saveWaveform;
cfg.debug.saveRoot = saveRoot;

pcbtest.acquire_debug(cfg);
