function [t, data, metadata] = load_waveform(filePath)
%LOAD_WAVEFORM Read a legacy or new waveform MAT file and validate it.
    if ~isfile(filePath)
        error('pcbtest:MissingWaveform', 'Waveform file not found: %s', filePath);
    end
    saved = load(filePath);
    if ~isfield(saved, 't')
        error('pcbtest:InvalidWaveform', 'Missing t in %s', filePath);
    end
    if isfield(saved, 'data')
        data = saved.data;
    elseif isfield(saved, 'data_read')
        data = saved.data_read;
    else
        error('pcbtest:InvalidWaveform', 'Missing data or data_read in %s', filePath);
    end
    t = saved.t;
    if ~isnumeric(t) || ~isnumeric(data) || ~isvector(t) || ...
            ~isvector(data) || numel(t) ~= numel(data) || numel(t) < 3
        error('pcbtest:InvalidWaveform', 'Invalid t/data dimensions in %s', filePath);
    end
    t = double(t(:));
    data = double(data(:));
    dt = diff(t);
    if any(~isfinite(t)) || any(~isfinite(data)) || any(dt <= 0) || ...
            max(abs(dt - dt(1))) > max(1e-12 * dt(1), 1e-15)
        error('pcbtest:InvalidWaveform', 'Invalid or nonuniform samples in %s', filePath);
    end
    metadata = struct();
    if isfield(saved, 'metadata')
        metadata = saved.metadata;
    end
end
