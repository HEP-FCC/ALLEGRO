function acquire_capacity(cfg)
%ACQUIRE_CAPACITY Capture one rising waveform per channel for offline fitting.
    if isempty(cfg.dataDir) || isempty(cfg.filePrefix) || ...
            isempty(cfg.hardware.matrixResource) || isempty(cfg.hardware.scopeResource)
        error('pcbtest:Config', ...
            'Set dataDir, filePrefix, and both VISA resource names before acquisition.');
    end
    channels = cfg.channels(:)';
    if isempty(channels) || any(channels < 1 | channels > 15 | ...
            channels ~= round(channels)) || numel(unique(channels)) ~= numel(channels)
        error('pcbtest:Config', 'cfg.channels must be unique integers from 1 to 15.');
    end
    if cfg.scope.readoutChannel == cfg.scope.referenceChannel
        error('pcbtest:Config', 'Reference and readout scope channels must differ.');
    end
    if ~isfolder(cfg.dataDir)
        mkdir(cfg.dataDir);
    end

    matrix = visadev(cfg.hardware.matrixResource);
    cleanupMatrix = onCleanup(@() write(matrix, 'C'));
    scope = visadev(cfg.hardware.scopeResource);
    matrixId = strtrim(writeread(matrix, '*IDN?'));
    scopeId = strtrim(writeread(scope, '*IDN?'));
    if ~isempty(cfg.hardware.matrixIdContains) && ...
            ~contains(matrixId, cfg.hardware.matrixIdContains)
        error('pcbtest:WrongInstrument', 'Unexpected matrix identity: %s', matrixId);
    end
    if ~isempty(cfg.hardware.scopeIdContains) && ...
            ~contains(scopeId, cfg.hardware.scopeIdContains)
        error('pcbtest:WrongInstrument', 'Unexpected scope identity: %s', scopeId);
    end
    fprintf('Matrix: %s\nScope: %s\n', matrixId, scopeId);
    flush(matrix);
    flush(scope);

    settings = cfg.capacity.scope;
    ch = cfg.scope.readoutChannel;
    for channel = channels
        name = sprintf('%s%d.mat', cfg.filePrefix, channel);
        filePath = fullfile(cfg.dataDir, name);
        if isfile(filePath) && cfg.hardware.skipExistingFiles
            [~, ~, oldMetadata] = pcbtest.load_waveform(filePath);
            if ~isfield(oldMetadata, 'channel') || ...
                    ~isfield(oldMetadata, 'tower') || ...
                    ~isfield(oldMetadata, 'scopeSettings') || ...
                    ~isfield(oldMetadata, 'scopeChannel') || ...
                    ~isfield(oldMetadata, 'matrixId') || ...
                    ~isfield(oldMetadata, 'scopeId') || ...
                    oldMetadata.channel ~= channel || ...
                    oldMetadata.tower ~= cfg.tower || ...
                    oldMetadata.scopeChannel ~= ch || ...
                    ~isequal(oldMetadata.scopeSettings, settings) || ...
                    ~strcmp(oldMetadata.matrixId, matrixId) || ...
                    ~strcmp(oldMetadata.scopeId, scopeId)
                error('pcbtest:ExistingFile', ...
                    'Existing file lacks matching metadata: %s', filePath);
            end
            fprintf('Skipping existing %s\n', name);
            continue;
        end

        write(matrix, 'C');
        pcbtest.set_matrix_channel(matrix, channel, 'inject');
        pcbtest.set_matrix_channel(matrix, channel, 'readout');
        write(scope, sprintf('CHANnel%d:POSition %g', ch, settings.position));
        write(scope, sprintf('CHANnel%d:RANGe %g', ch, settings.range));
        write(scope, sprintf('CHANnel%d:OFFset %g', ch, settings.offset));
        pause(cfg.capacity.settleSeconds);
        write(scope, 'SING');
        completion = strtrim(writeread(scope, '*OPC?'));
        if ~strcmp(completion, '1')
            error('pcbtest:ScopeCompletion', ...
                'Scope did not confirm acquisition for %s: %s', name, completion);
        end
        if cfg.scope.readDelaySeconds > 0
            pause(cfg.scope.readDelaySeconds);
        end
        [t, data_read] = pcbtest.read_scope(scope, ch);
        metadata = struct('acquiredAt', char(datetime('now')), ...
            'measurement', 'capacity', 'tower', cfg.tower, ...
            'channel', channel, 'scopeChannel', ch, ...
            'referenceScopeChannel', cfg.scope.referenceChannel, ...
            'matrixId', matrixId, 'scopeId', scopeId, ...
            'scopeSettings', settings);
        save(filePath, 't', 'data_read', 'metadata', '-mat');
        fprintf('Saved %s\n', name);
    end
    clear cleanupMatrix;
end
