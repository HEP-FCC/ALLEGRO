function acquire_debug(cfg)
%ACQUIRE_DEBUG Capture one routed pair and optionally save its waveform.
    settings = cfg.debug.scopeSettings;
    injectChannel = cfg.debug.injectChannel;
    readoutChannel = cfg.debug.readoutChannel;
    scopeChannel = cfg.scope.readoutChannel;
    if isempty(cfg.hardware.matrixResource) || isempty(cfg.hardware.scopeResource)
        error('pcbtest:Config', 'Set both VISA resource names before debugging.');
    end
    if scopeChannel == cfg.scope.referenceChannel
        error('pcbtest:Config', 'Reference and readout scope channels must differ.');
    end
    if ~isscalar(injectChannel) || ~isscalar(readoutChannel) || ...
            ~ismember(injectChannel, 1:15) || ~ismember(readoutChannel, 1:15)
        error('pcbtest:Config', 'Injection and readout channels must be 1:15.');
    end
    if ~isscalar(scopeChannel) || ~isfinite(scopeChannel) || ...
            scopeChannel < 1 || scopeChannel ~= round(scopeChannel)
        error('pcbtest:Config', 'scopeChannel must be a positive integer.');
    end
    if ~isscalar(cfg.debug.settleSeconds) || ...
            ~isfinite(cfg.debug.settleSeconds) || cfg.debug.settleSeconds < 0
        error('pcbtest:Config', 'settleSeconds must be nonnegative.');
    end
    if cfg.debug.applyScopeSettings && ...
            (~isfinite(settings.position) || ~isfinite(settings.range) || ...
            settings.range <= 0 || ~isfinite(settings.offset))
        error('pcbtest:Config', 'Scope position, range, and offset must be valid.');
    end

    fprintf('\nDebug acquisition: T%d, injection %d, readout %d.\n', ...
        cfg.tower, injectChannel, readoutChannel);

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

    write(matrix, 'C');
    pcbtest.set_matrix_channel(matrix, injectChannel, 'inject');
    pcbtest.set_matrix_channel(matrix, readoutChannel, 'readout');
    if cfg.debug.applyScopeSettings
        write(scope, sprintf('CHANnel%d:POSition %g', ...
            scopeChannel, settings.position));
        write(scope, sprintf('CHANnel%d:RANGe %g', ...
            scopeChannel, settings.range));
        write(scope, sprintf('CHANnel%d:OFFset %g', ...
            scopeChannel, settings.offset));
        fprintf('CH%d settings: position=%g, range=%g, offset=%g\n', ...
            scopeChannel, settings.position, settings.range, settings.offset);
    else
        fprintf('CH%d vertical settings remain as set on the scope.\n', scopeChannel);
    end
    if cfg.debug.settleSeconds > 0
        pause(cfg.debug.settleSeconds);
    end
    write(scope, 'SING');
    completion = strtrim(writeread(scope, '*OPC?'));
    if ~strcmp(completion, '1')
        error('pcbtest:ScopeCompletion', ...
            'Scope did not confirm the debug acquisition: %s', completion);
    end
    if cfg.scope.readDelaySeconds > 0
        pause(cfg.scope.readDelaySeconds);
    end
    [t, data] = pcbtest.read_scope(scope, scopeChannel);

    fprintf('Acquisition complete. Inspect the waveform on the scope.\n');
    if ~cfg.debug.saveWaveform
        fprintf('Waveform not saved.\n');
        return;
    end

    metadata = struct('acquiredAt', char(datetime('now')), ...
        'measurement', 'debug', 'tower', cfg.tower, ...
        'injectChannel', injectChannel, 'readoutChannel', readoutChannel, ...
        'scopeChannel', scopeChannel, 'matrixId', matrixId, ...
        'referenceScopeChannel', cfg.scope.referenceChannel, ...
        'scopeId', scopeId, ...
        'scopeSettingsApplied', cfg.debug.applyScopeSettings);
    if cfg.debug.applyScopeSettings
        metadata.scopeSettings = settings;
    else
        metadata.scopeSettings = struct(); % Front-panel settings are unknown.
    end
    saveDir = fullfile(cfg.debug.saveRoot, sprintf('T%d', cfg.tower));
    if ~isfolder(saveDir)
        mkdir(saveDir);
    end
    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'));
    fileName = sprintf('T%d_debug_%02d-%02d_%s.mat', ...
        cfg.tower, injectChannel, readoutChannel, timestamp);
    filePath = fullfile(saveDir, fileName);
    if isfile(filePath)
        error('pcbtest:ExistingFile', 'Debug file already exists: %s', filePath);
    end
    save(filePath, 't', 'data', 'metadata', '-mat');
    fprintf('Saved %s\n', filePath);
    clear cleanupMatrix;
end
