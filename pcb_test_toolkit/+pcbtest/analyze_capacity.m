function result = analyze_capacity(cfg)
%ANALYZE_CAPACITY Calculate per-channel capacitance relative to tower T0.
    required = {'tower', 'dataDir', 'outputDir', 'filePrefix', ...
        'displayOrder', 'displayOmitPosition', 'capacity'};
    for k = 1:numel(required)
        if ~isfield(cfg, required{k})
            error('pcbtest:Config', 'Missing configuration field: %s', required{k});
        end
    end
    if isempty(cfg.outputDir)
        error('pcbtest:Config', 'Set cfg.outputDir.');
    end
    if ~isequal(sort(cfg.displayOrder(:))', 1:15) || ...
            ~ismember(cfg.displayOmitPosition, 1:15)
        error('pcbtest:Config', 'Display order must contain channels 1:15 once.');
    end
    if ~isfinite(cfg.capacity.resistanceOhm) || cfg.capacity.resistanceOhm <= 0
        error('pcbtest:Config', 'Capacity resistance must be positive.');
    end

    positions = setdiff(1:15, cfg.displayOmitPosition, 'stable');
    displayChannels = cfg.displayOrder(positions);

    if strcmp(cfg.capacity.source, 'waveforms')
        if ~isfolder(cfg.dataDir) || isempty(cfg.filePrefix)
            error('pcbtest:Config', 'Set capacity waveform dataDir and filePrefix.');
        end
        tauSeconds = fitWaveformSet(cfg.dataDir, cfg.filePrefix, cfg.capacity);
        if isempty(cfg.capacity.referenceTauSeconds)
            if ~isfolder(cfg.capacity.referenceDir) || ...
                    isempty(cfg.capacity.referencePrefix)
                error('pcbtest:Config', 'Set the T0 reference directory and prefix.');
            end
            referenceTauSeconds = fitWaveformSet( ...
                cfg.capacity.referenceDir, cfg.capacity.referencePrefix, cfg.capacity);
        else
            referenceTauSeconds = cfg.capacity.referenceTauSeconds(:)';
        end
    elseif strcmp(cfg.capacity.source, 'tauFiles')
        if ~isfolder(cfg.capacity.tauDir)
            error('pcbtest:Config', 'Set cfg.capacity.tauDir.');
        end
        tauSeconds = loadTau(fullfile(cfg.capacity.tauDir, ...
            sprintf(cfg.capacity.tauPattern, cfg.tower)));
        referenceTauSeconds = loadTau(fullfile(cfg.capacity.tauDir, ...
            sprintf(cfg.capacity.tauPattern, 0)));
    else
        error('pcbtest:Config', 'Capacity source must be waveforms or tauFiles.');
    end
    if numel(tauSeconds) ~= 15 || numel(referenceTauSeconds) ~= 15
        error('pcbtest:CapacityData', 'Capacity tau arrays must contain 15 values.');
    end
    % Invalid measurement or reference tau makes that channel unavailable.
    tauSeconds(~isfinite(tauSeconds) | tauSeconds <= 0) = NaN;
    referenceTauSeconds(~isfinite(referenceTauSeconds) | ...
        referenceTauSeconds <= 0) = NaN;

    capacitancePf = (tauSeconds - referenceTauSeconds) / ...
        cfg.capacity.resistanceOhm * 1e12;
    displayValues = capacitancePf(displayChannels);
    invalidChannels = find(~isfinite(capacitancePf));
    if ~isempty(invalidChannels)
        warning('pcbtest:InvalidCapacityChannels', ...
            ['Tower T%d: capacitance is NaN for acquisition channels %s. ' ...
            'Invalid measurement tau: %s; invalid T0 tau: %s.'], ...
            cfg.tower, mat2str(invalidChannels), ...
            mat2str(find(isnan(tauSeconds))), ...
            mat2str(find(isnan(referenceTauSeconds))));
    end
    result = struct('tower', cfg.tower, ...
        'tauSeconds', tauSeconds, ...
        'referenceTauSeconds', referenceTauSeconds, ...
        'capacitancePf', capacitancePf, ...
        'resistanceOhm', cfg.capacity.resistanceOhm, ...
        'displayChannels', displayChannels, ...
        'displayPositions', positions, ...
        'displayCapacitancePf', displayValues, ...
        'invalidChannels', invalidChannels);

    if cfg.capacity.fitComponents
        lineLength = cfg.capacity.lineLengthMm(:);
        couplingCount = cfg.capacity.couplingCount(:);
        nDisplay = numel(displayValues);
        if numel(lineLength) ~= nDisplay || numel(couplingCount) ~= nDisplay
            error('pcbtest:CapacityGeometry', ...
                'Line lengths and coupling counts must match the display cells.');
        end
        fitMask = isfinite(displayValues(:));
        geometry = [lineLength, couplingCount];
        result.componentFitAvailable = rank(geometry(fitMask, :)) == 2;
        if result.componentFitAvailable
            coefficients = geometry(fitMask, :) \ displayValues(fitMask)';
        else
            coefficients = [NaN; NaN];
            warning('pcbtest:InsufficientCapacityData', ...
                ['Tower T%d: valid cells do not support a two-component fit. ' ...
                'Measured capacitances are exported without a component fit.'], ...
                cfg.tower);
        end
        result.lineicPfPerMm = coefficients(1);
        result.couplingPfPerCount = coefficients(2);
        result.lineContributionPf = lineLength * coefficients(1);
        result.couplingContributionPf = couplingCount * coefficients(2);
        result.fittedCapacitancePf = geometry * coefficients;
        result.fitIncluded = fitMask & result.componentFitAvailable;
    end

    if ~isfolder(cfg.outputDir)
        mkdir(cfg.outputDir);
    end
    tag = sprintf('T%d', cfg.tower);
    save(fullfile(cfg.outputDir, ['capacity_' tag '.mat']), 'result', '-mat');
    summary = table((1:15)', tauSeconds(:), referenceTauSeconds(:), ...
        capacitancePf(:), 'VariableNames', ...
        {'channel', 'tauSeconds', 'referenceTauSeconds', 'capacitancePf'});
    writetable(summary, fullfile(cfg.outputDir, ['capacity_' tag '.csv']));

    fig = figure('Visible', 'off');
    % The historical plain plot uses acquisition order; the fit uses cell order.
    plainValues = capacitancePf([1:7, 9:15]);
    xEdges = 0.5:1:(numel(plainValues) + 0.5);
    stairs(xEdges, [plainValues(:); plainValues(end)]);
    xlim([0.5, numel(plainValues) + 0.5]);
    ylim(cfg.capacity.plotYLimitPf);
    xticks(1:numel(plainValues));
    ax = gca;
    ax.YAxis.MinorTick = 'on';
    xlabel('Cell');
    ylabel('Capacity [pF]');
    title(sprintf('Capacity of Each Cell in T%d', cfg.tower));
    exportgraphics(fig, fullfile(cfg.outputDir, ...
        ['capacity_' tag '.pdf']), 'ContentType', 'vector');
    close(fig);

    if cfg.capacity.fitComponents
        fitSummary = table((1:numel(displayValues))', displayChannels(:), ...
            displayValues(:), result.lineContributionPf, ...
            result.couplingContributionPf, result.fittedCapacitancePf, ...
            result.fitIncluded, 'VariableNames', ...
            {'displayCell', 'acquisitionChannel', 'measuredPf', ...
            'lineContributionPf', 'couplingContributionPf', ...
            'fittedPf', 'includedInFit'});
        writetable(fitSummary, fullfile(cfg.outputDir, ...
            [tag '_capacity_fit.csv']));
        % Keep the historical font sizes, with a larger canvas for margins.
        fitFig = figure('Visible', 'off', 'Color', 'white', ...
            'Units', 'pixels', 'Position', [100, 100, 720, 540]);
        plot(1:numel(displayValues), displayValues, 'ko', ...
            'LineWidth', 1.5, 'MarkerSize', 8);
        hold on;
        if result.componentFitAvailable
            stairs(xEdges, [result.lineContributionPf; result.lineContributionPf(end)], ...
                'k--', 'LineWidth', 1.6);
            stairs(xEdges, [result.couplingContributionPf; ...
                result.couplingContributionPf(end)], 'b:', 'LineWidth', 2.0);
            stairs(xEdges, [result.fittedCapacitancePf; ...
                result.fittedCapacitancePf(end)], 'r-', 'LineWidth', 2.0);
            legendHandle = legend('Measurement', 'Lineic Capacitance', ...
                'Coupling Capacitance', 'Total');
        else
            legendHandle = legend('Measurement');
        end
        legendHandle.Location = 'northwest';
        xlim([0.5, numel(displayValues) + 0.5]);
        ylim(cfg.capacity.fitPlotYLimitPf);
        xticks(1:numel(displayValues));
        ax = gca;
        ax.FontSize = 18;
        ax.YAxis.MinorTick = 'on';
        lineText = ['$dC/d\ell$ = ', ...
            num2str(result.lineicPfPerMm, '%.2f'), ' pF/mm'];
        couplingText = ['$dC/dn$ = ', ...
            num2str(result.couplingPfPerCount, '%.2f'), ' pF'];
        if ~result.componentFitAvailable
            lineText = 'Insufficient valid cells for component fit';
            couplingText = '';
        end
        text(0.025, 0.600, lineText, 'Interpreter', 'latex', ...
            'Units', 'normalized', 'FontSize', 18, ...
            'VerticalAlignment', 'bottom');
        text(0.025, 0.525, couplingText, 'Interpreter', 'latex', ...
            'Units', 'normalized', 'FontSize', 18, ...
            'VerticalAlignment', 'bottom');
        xlabel('Cell');
        ylabel('Capacitance [pF]');
        title(sprintf('Capacitance of Each Cell in T%d', cfg.tower));
        legendHandle.FontSize = 18;
        hold off;
        % Leave 24 pixels beyond the title, tick labels, and axis labels.
        % Repeat after layout because text extents can change with axes size.
        ax.Units = 'pixels';
        padding = 24;
        for layoutPass = 1:2
            drawnow;
            inset = ax.TightInset;
            canvasSize = fitFig.Position(3:4);
            ax.Position = [padding + inset(1), padding + inset(2), ...
                canvasSize(1) - 2 * padding - inset(1) - inset(3), ...
                canvasSize(2) - 2 * padding - inset(2) - inset(4)];
        end
        drawnow;
        % Export the full canvas so PDF cropping cannot discard the margins.
        fitFig.Units = 'inches';
        pageSize = fitFig.Position(3:4);
        fitFig.PaperUnits = 'inches';
        fitFig.PaperSize = pageSize;
        fitFig.PaperPosition = [0, 0, pageSize];
        print(fitFig, fullfile(cfg.outputDir, ...
            [tag '_capacity_fit.pdf']), '-dpdf', '-painters');
        close(fitFig);
    end
    fprintf('Analyzed capacity for tower T%d.\n', cfg.tower);
end

function tauSeconds = fitWaveformSet(folder, prefix, settings)
    tauSeconds = zeros(1, 15);
    for channel = 1:15
        filePath = fullfile(folder, sprintf('%s%d.mat', prefix, channel));
        [t, data] = pcbtest.load_waveform(filePath);
        try
            tauSeconds(channel) = pcbtest.fit_tau(t, data, settings);
        catch exception
            if strcmp(exception.identifier, 'pcbtest:CapacityFit')
                tauSeconds(channel) = NaN;
                warning('pcbtest:InvalidCapacityWaveform', ...
                    ['Channel %d has no valid tau and is recorded as NaN. ' ...
                    'File: %s. Reason: %s'], ...
                    channel, filePath, exception.message);
                continue;
            end
            error('pcbtest:CapacityFit', ...
                'Cannot fit %s: %s', filePath, exception.message);
        end
    end
end

function tauSeconds = loadTau(filePath)
    if ~isfile(filePath)
        error('pcbtest:MissingTau', 'Tau file not found: %s', filePath);
    end
    saved = load(filePath, 'tau_array');
    if ~isfield(saved, 'tau_array') || ~isnumeric(saved.tau_array) || ...
            numel(saved.tau_array) ~= 15
        error('pcbtest:InvalidTau', 'Expected 15 tau_array values in %s', filePath);
    end
    tauSeconds = double(saved.tau_array(:)');
end
