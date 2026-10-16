% Acquire the 15 x 15 cross-talk matrix for one physically connected tower.
toolkitDir = fileparts(mfilename('fullpath'));
addpath(toolkitDir);

% Edit these settings for this cross-talk measurement campaign.
tower = 4;                     % Change this after manually connecting another tower.
dataRoot = fullfile(toolkitDir, 'data', 'xtalk_run_001');
datasetLabel = 'PCBCHINA';
skipExistingFiles = false;      % false reacquires and overwrites existing pair files.
scopeSignalRange = 5;           % SCPI CHANnel:RANGe for inject == readout.
scopeAdjacentRange = 0.3;       % SCPI CHANnel:RANGe for channel numbers one apart.
scopeOtherRange = 0.1;          % SCPI CHANnel:RANGe for all other pairs.
scopeSignalPosition = 0;
scopeAdjacentPosition = 0;
scopeOtherPosition = 0;
scopeSignalOffset = 0;
scopeAdjacentOffset = 0;
scopeOtherOffset = 0;
readoutScopeChannel = 2;        % CH1 is the untouched generator reference.
readDelaySeconds = 0.5;        % Wait after the single acquisition before reading CH2.
matrixResource = 'GPIB0::7::INSTR';
scopeResource = 'USB0::0x0AAD::0x01D6::105097::0::INSTR';
matrixIdContains = '';          % Optional expected text from *IDN?.
scopeIdContains = '';           % Optional expected text from *IDN?.

if isempty(matrixResource) || isempty(scopeResource)
    error('pcbtest:Config', 'Set both VISA resource names before acquisition.');
end

cfg = pcbtest.default_config(tower);
cfg.dataDir = fullfile(dataRoot, sprintf('T%d', tower));
cfg.filePrefix = sprintf('T%d_xtalk_%s_', tower, datasetLabel);
cfg.channels = 1:15;
cfg.hardware.matrixResource = matrixResource;
cfg.hardware.scopeResource = scopeResource;
cfg.hardware.matrixIdContains = matrixIdContains;
cfg.hardware.scopeIdContains = scopeIdContains;
cfg.hardware.skipExistingFiles = skipExistingFiles;
cfg.scope.signal.range = scopeSignalRange;
cfg.scope.adjacent.range = scopeAdjacentRange;
cfg.scope.other.range = scopeOtherRange;
cfg.scope.signal.position = scopeSignalPosition;
cfg.scope.adjacent.position = scopeAdjacentPosition;
cfg.scope.other.position = scopeOtherPosition;
cfg.scope.signal.offset = scopeSignalOffset;
cfg.scope.adjacent.offset = scopeAdjacentOffset;
cfg.scope.other.offset = scopeOtherOffset;
cfg.scope.readoutChannel = readoutScopeChannel;
cfg.scope.readDelaySeconds = readDelaySeconds;

fprintf('\nCross-talk: connect tower T%d before acquisition.\n', tower);
reply = strtrim(input('Press Enter when ready, or type q to stop: ', 's'));
if strcmpi(reply, 'q')
    fprintf('Stopped before tower T%d.\n', tower);
    return;
elseif ~isempty(reply)
    error('pcbtest:OperatorInput', 'Press Enter or type q.');
end

pcbtest.acquire_xtalk(cfg);
fprintf('Completed tower T%d. Data: %s\n', tower, cfg.dataDir);
