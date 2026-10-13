function [t, data] = read_scope(scope, channel)
%READ_SCOPE Read an ASCII waveform using the historical scope SCPI dialect.
    headerText = writeread(scope, sprintf('CHAN%d:DATA:HEAD?', channel));
    header = str2double(strsplit(strtrim(headerText), ','));
    if numel(header) < 3 || any(~isfinite(header(1:3))) || ...
            header(3) < 3 || header(3) ~= round(header(3))
        error('pcbtest:ScopeHeader', 'Invalid scope waveform header: %s', headerText);
    end
    writeline(scope, ':FORMAT ASCii');
    writeline(scope, sprintf('CHAN%d:DATA?', channel));
    raw = readline(scope);
    data = str2double(strsplit(strtrim(raw), ','));
    if numel(data) ~= header(3) || any(~isfinite(data))
        error('pcbtest:ScopeData', ...
            'Scope returned %d samples; header specifies %d.', numel(data), header(3));
    end
    t = linspace(header(1), header(2), header(3));
end
