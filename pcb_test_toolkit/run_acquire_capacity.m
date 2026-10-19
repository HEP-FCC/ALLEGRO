% Acquire one capacity waveform per channel for one physically connected tower.
toolkitDir = fileparts(mfilename('fullpath'));
addpath(toolkitDir);

% Edit these settings for this capacity measurement campaign.
tower = 0;                     % Acquire T0 first, then change this after rewiring.
dataRoot = fullfile(toolkitDir, 'data', 'capacity_run_001');
datasetLabel = 'PCBCHINA';
skipExistingFiles = true;       % false reacquires and overwrites existing channel files.
scopeRange = 2;                 % SCPI CHANnel:RANGe for every capacity channel.
readoutScopeChannel = 2;        % CH1 is the untouched generator reference.
readDelaySeconds = 1;          % Wait after the single acquisition before reading CH2.
matrixResource = 'GPIB0::7::INSTR';
scopeResource = 'USB0::0x0AAD::0x01D6::105097::0::INSTR';
matrixIdContains = '';          % Optional expected text from *IDN?.
scopeIdContains = '';           % Optional expected text from *IDN?.

if isempty(matrixResource) || isempty(scopeResource)
    error('pcbtest:Config', 'Set both VISA resource names before acquisition.');
end

cfg = pcbtest.default_config(tower);
cfg.dataDir = fullfile(dataRoot, sprintf('T%d', tower));
cfg.filePrefix = sprintf('T%d_Capacity_%s_', tower, datasetLabel);
cfg.channels = 1:15;
cfg.hardware.matrixResource = matrixResource;
cfg.hardware.scopeResource = scopeResource;
cfg.hardware.matrixIdContains = matrixIdContains;
cfg.hardware.scopeIdContains = scopeIdContains;
cfg.hardware.skipExistingFiles = skipExistingFiles;
cfg.capacity.scope.range = scopeRange;
cfg.scope.readoutChannel = readoutScopeChannel;
cfg.scope.readDelaySeconds = readDelaySeconds;

fprintf('\nCapacity: connect tower T%d before acquisition.\n', tower);
reply = strtrim(input('Press Enter when ready, or type q to stop: ', 's'));
if strcmpi(reply, 'q')
    fprintf('Stopped before tower T%d.\n', tower);
    return;
elseif ~isempty(reply)
    error('pcbtest:OperatorInput', 'Press Enter or type q.');
end

pcbtest.acquire_capacity(cfg);
fprintf('Completed capacity tower T%d. Data: %s\n', tower, cfg.dataDir);
