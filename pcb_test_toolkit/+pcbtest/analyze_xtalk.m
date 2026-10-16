function results = analyze_xtalk(cfg)
%ANALYZE_XTALK Analyze a complete 15 x 15 cross-talk measurement offline.
    required = {'tower', 'dataDir', 'outputDir', 'filePrefix', ...
        'shapingTimes', 'baselineFraction', 'displayOrder', 'displayOmitPosition'};
    for k = 1:numel(required)
        if ~isfield(cfg, required{k})
            error('pcbtest:Config', 'Missing configuration field: %s', required{k});
        end
    end
    if ~isfolder(cfg.dataDir)
        error('pcbtest:Config', 'Data folder not found: %s', cfg.dataDir);
    end
    if isempty(cfg.outputDir) || isempty(cfg.filePrefix)
        error('pcbtest:Config', 'Set cfg.outputDir and cfg.filePrefix.');
    end
    if ~isequal(sort(cfg.displayOrder(:))', 1:15) || ...
            ~ismember(cfg.displayOmitPosition, 1:15)
        error('pcbtest:Config', 'Display order must contain channels 1:15 once.');
    end
    if cfg.baselineFraction <= 0 || cfg.baselineFraction > 1 || ...
            isempty(cfg.shapingTimes) || any(~isfinite(cfg.shapingTimes)) || ...
            any(cfg.shapingTimes <= 0)
        error('pcbtest:Config', 'Invalid baseline fraction or shaping times.');
    end

    n = 15;
    paths = cell(n, n);
    referenceT = [];
    for inject = 1:n
        for readout = 1:n
            name = sprintf('%s%02d-%02d.mat', cfg.filePrefix, inject, readout);
            paths{inject, readout} = fullfile(cfg.dataDir, name);
            [t, ~] = pcbtest.load_waveform(paths{inject, readout});
            if isempty(referenceT)
                referenceT = t;
            elseif numel(t) ~= numel(referenceT) || ...
                    max(abs(t - referenceT)) > max(1e-12 * (t(2)-t(1)), 1e-15)
                error('pcbtest:TimeAxis', 'Time axis differs in %s', paths{inject, readout});
            end
        end
    end

    if ~isfolder(cfg.outputDir)
        mkdir(cfg.outputDir);
    end
    sampleInterval = referenceT(2) - referenceT(1);
    keepPositions = setdiff(1:n, cfg.displayOmitPosition, 'stable');
    displayChannels = cfg.displayOrder(keepPositions);
    results = struct([]);

    for timeIndex = 1:numel(cfg.shapingTimes)
        shapingTime = cfg.shapingTimes(timeIndex);
        rawMax = zeros(n, n);
        rawAtInjectionPeak = zeros(n, n);
        peakIndex = zeros(n, 1);

        % Derive each reference peak from its diagonal (injected) waveform.
        for inject = 1:n
            [~, data] = pcbtest.load_waveform(paths{inject, inject});
            shaped = shapeOne(data, sampleInterval, shapingTime, cfg.baselineFraction);
            [~, peakIndex(inject)] = max(abs(shaped));
        end

        for inject = 1:n
            for readout = 1:n
                [~, data] = pcbtest.load_waveform(paths{inject, readout});
                shaped = shapeOne(data, sampleInterval, shapingTime, cfg.baselineFraction);
                rawMax(inject, readout) = max(abs(shaped));
                rawAtInjectionPeak(inject, readout) = abs(shaped(peakIndex(inject)));
            end
        end

        denominator = max(rawMax(:));
        if denominator <= 0
            error('pcbtest:ZeroSignal', 'All shaped waveforms are zero.');
        end
        relativeGlobal = rawMax / denominator;
        displayedFull = relativeGlobal(cfg.displayOrder, cfg.displayOrder);
        displayedFull(1:n+1:end) = 0;
        displayMatrix = displayedFull(keepPositions, keepPositions);
        offDiagonal = ~eye(size(displayMatrix));

        result = struct();
        result.tower = cfg.tower;
        result.filePrefix = cfg.filePrefix;
        result.shapingTimeSeconds = shapingTime;
        result.baselineFraction = cfg.baselineFraction;
        result.rawMaxV = rawMax;
        result.rawAtInjectionPeakV = rawAtInjectionPeak;
        result.relativeGlobal = relativeGlobal;
        result.globalDenominatorV = denominator;
        result.displayMatrix = displayMatrix;
        result.displayChannels = displayChannels;
        result.displayPositions = keepPositions;
        peakFull = rawAtInjectionPeak(cfg.displayOrder, cfg.displayOrder);
        peakFull(1:n+1:end) = 0;
        result.displayPeakMatrixV = peakFull(keepPositions, keepPositions);
        result.maxDisplayed = max(displayMatrix(:));
        result.meanDisplayedOffDiagonal = mean(displayMatrix(offDiagonal));
        if timeIndex == 1
            results = result;
        else
            results(timeIndex) = result;
        end

        tag = sprintf('T%d_%gns', cfg.tower, shapingTime * 1e9);
        tag = strrep(tag, '.', 'p');
        save(fullfile(cfg.outputDir, ['results_' tag '.mat']), 'result', '-mat');

        [injectGrid, readoutGrid] = ndgrid(1:n, 1:n);
        pairs = table(injectGrid(:), readoutGrid(:), rawMax(:), ...
            rawAtInjectionPeak(:), relativeGlobal(:), ...
            'VariableNames', {'injectChannel', 'readoutChannel', ...
            'rawMaxV', 'rawAtInjectionPeakV', 'relativeGlobal'});
        writetable(pairs, fullfile(cfg.outputDir, ['pairs_' tag '.csv']));

        colorLimit = 6e-3;
        limitIndex = find(cfg.xtalkColorLimits(:, 1) == shapingTime, 1);
        if ~isempty(limitIndex)
            colorLimit = cfg.xtalkColorLimits(limitIndex, 2);
        end
        timeLabel = sprintf('%gns', shapingTime * 1e9);
        towerLabel = sprintf('T%d', cfg.tower);
        mainPlotPath = fullfile(cfg.outputDir, ...
            [towerLabel '_xtalk_' timeLabel '_plot.pdf']);
        plotHeatmap(displayMatrix, colorLimit, ...
            sprintf('Relative Cross-talk, Tower %d, \\tau = %g ns', ...
            cfg.tower, shapingTime * 1e9), ...
            'Readout cell', 'Injection cell', ...
            mainPlotPath, true, false);
        plotHeatmap(result.displayPeakMatrixV, [], ...
            sprintf('xtalk at signal peak (V) with Tsh=%s of %s', ...
            timeLabel, towerLabel), ...
            'readout cell ID', 'injection cell ID', ...
            fullfile(cfg.outputDir, ...
            [towerLabel '_peak_xtalk_' timeLabel '_plot.pdf']), ...
            false, true);
        fprintf('Analyzed tower %d at %g ns: max displayed = %.6g\n', ...
            cfg.tower, shapingTime * 1e9, result.maxDisplayed);
    end
end

function plotHeatmap(values, colorLimit, plotTitle, xLabel, yLabel, ...
        filePath, squareAxes, showGrid)
% Reproduce the original turbo heatmaps and vector PDF output.
    fig = figure('Visible', 'off');
    imagesc(values);
    if squareAxes
        axis square;
    end
    if showGrid
        grid on;
    end
    colorbar;
    colormap(fig, 'turbo');
    if ~isempty(colorLimit)
        clim([0, colorLimit]);
    end
    nCells = size(values, 1);
    set(gca, 'XTick', 1:nCells, 'YTick', 1:nCells);
    xlabel(xLabel);
    ylabel(yLabel);
    title(plotTitle);
    exportgraphics(fig, filePath, 'ContentType', 'vector');
    close(fig);
end

function shaped = shapeOne(data, sampleInterval, shapingTime, baselineFraction)
% Keep baseline subtraction identical to the historical analysis.
    nBaseline = max(1, round(numel(data) * baselineFraction));
    corrected = data - mean(data(1:nBaseline));
    shaped = pcbtest.shape_waveform(corrected, sampleInterval, shapingTime);
end
