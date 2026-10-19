function acquire_xtalk(cfg)
%ACQUIRE_XTALK Capture cross-talk waveforms on the Windows instrument PC.
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
    checkIdentity(matrixId, cfg.hardware.matrixIdContains, 'matrix');
    checkIdentity(scopeId, cfg.hardware.scopeIdContains, 'scope');
    fprintf('Matrix: %s\nScope: %s\n', matrixId, scopeId);
    flush(matrix);
    flush(scope);

    for inject = channels
        for readout = channels
            name = sprintf('%s%02d-%02d.mat', cfg.filePrefix, inject, readout);
            filePath = fullfile(cfg.dataDir, name);
            if inject == readout
                settings = cfg.scope.signal;
            elseif abs(inject - readout) == 1
                settings = cfg.scope.adjacent;
            else
                settings = cfg.scope.other;
            end
            ch = cfg.scope.readoutChannel;
            if isfile(filePath) && cfg.hardware.skipExistingFiles
                [~, ~, oldMetadata] = pcbtest.load_waveform(filePath);
                if ~isfield(oldMetadata, 'injectChannel') || ...
                        ~isfield(oldMetadata, 'readoutChannel') || ...
                        ~isfield(oldMetadata, 'tower') || ...
                        ~isfield(oldMetadata, 'scopeSettings') || ...
                        ~isfield(oldMetadata, 'scopeChannel') || ...
                        ~isfield(oldMetadata, 'matrixId') || ...
                        ~isfield(oldMetadata, 'scopeId') || ...
                        oldMetadata.injectChannel ~= inject || ...
                        oldMetadata.readoutChannel ~= readout || ...
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
            pcbtest.set_matrix_channel(matrix, inject, 'inject');
            pcbtest.set_matrix_channel(matrix, readout, 'readout');
            write(scope, sprintf('CHANnel%d:POSition %g', ch, settings.position));
            write(scope, sprintf('CHANnel%d:RANGe %g', ch, settings.range));
            write(scope, sprintf('CHANnel%d:OFFset %g', ch, settings.offset));
            if cfg.hardware.settleSeconds > 0
                pause(cfg.hardware.settleSeconds);
            end
            write(scope, 'SING');
            completion = strtrim(writeread(scope, '*OPC?'));
            if ~strcmp(completion, '1')
                error('pcbtest:ScopeCompletion', ...
                    'Scope did not confirm acquisition for %s: %s', name, completion);
            end
            if cfg.scope.readDelaySeconds > 0
                pause(cfg.scope.readDelaySeconds);
            end
            [t, data] = pcbtest.read_scope(scope, ch);
            metadata = struct('acquiredAt', char(datetime('now')), ...
                'tower', cfg.tower, 'injectChannel', inject, ...
                'readoutChannel', readout, 'scopeChannel', ch, ...
                'referenceScopeChannel', cfg.scope.referenceChannel, ...
                'matrixId', matrixId, 'scopeId', scopeId, ...
                'scopeSettings', settings);
            save(filePath, 't', 'data', 'metadata', '-mat');
            fprintf('Saved %s\n', name);
        end
    end
    clear cleanupMatrix;
end

function checkIdentity(actual, expected, deviceName)
    if ~isempty(expected) && ~contains(actual, expected)
        error('pcbtest:WrongInstrument', ...
            '%s ID "%s" does not contain expected text "%s".', ...
            deviceName, actual, expected);
    end
end
