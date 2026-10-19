function shaped = shape_waveform(data, sampleInterval, shapingTime)
%SHAPE_WAVEFORM Apply the historical third-order CR-RC shaping response.
    if ~isfinite(sampleInterval) || sampleInterval <= 0 || ...
            ~isfinite(shapingTime) || shapingTime <= 0
        error('pcbtest:InvalidShaping', 'Sample interval and shaping time must be positive.');
    end
    persistent cachedInterval cachedShapingTime b a
    if isempty(cachedInterval) || sampleInterval ~= cachedInterval || ...
            shapingTime ~= cachedShapingTime
        denominator = conv(conv([shapingTime, 1], [shapingTime, 1]), ...
            [shapingTime, 1]);
        analog = tf([shapingTime, 0], denominator);
        discrete = c2d(analog, sampleInterval, 'tustin');
        [b, a] = tfdata(discrete, 'v');
        cachedInterval = sampleInterval;
        cachedShapingTime = shapingTime;
    end
    shaped = filter(b, a, data);
end
