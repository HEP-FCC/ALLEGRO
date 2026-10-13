function [tauSeconds, amplitudeV, startTimeSeconds] = fit_tau(t, data, settings)
%FIT_TAU Fit the legacy rising exponential to one capacity waveform.
    if numel(t) ~= numel(data) || numel(t) < 3
        error('pcbtest:CapacityFit', 'Time and waveform vectors must have equal length.');
    end
    t = double(t(:));
    corrected = double(data(:)) - double(data(1));
    startIndex = find(corrected > settings.thresholdV, 1, 'first');
    if isempty(startIndex)
        error('pcbtest:CapacityFit', ...
            'Waveform never crosses the %.4g V fit threshold.', settings.thresholdV);
    end
    startTimeSeconds = t(startIndex);
    fitTime = t(startIndex:end) - startTimeSeconds;
    fitData = corrected(startIndex:end);
    model = @(parameters, time) parameters(1) .* ...
        (1 - exp(-time ./ parameters(2)));
    initial = [settings.initialAmplitudeV, settings.initialTauSeconds];
    lower = [0, eps];
    options = optimoptions('lsqcurvefit', 'Display', 'off');
    fitted = lsqcurvefit(model, initial, fitTime, fitData, ...
        lower, [], options);
    amplitudeV = fitted(1);
    tauSeconds = fitted(2);
    if ~isfinite(tauSeconds) || tauSeconds <= 0
        error('pcbtest:CapacityFit', 'The fitted time constant is invalid.');
    end
end
