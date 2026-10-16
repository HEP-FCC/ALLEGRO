function plot_xtalk_study(cfg)
%PLOT_XTALK_STUDY Plot all readout waveforms for selected injection cells.
% The first view offsets and high-pass filters the cross-talk traces as in
% the historical study script. The second view overlays the shaped traces.
    if isempty(cfg.studyCells)
        return;
    end
    if any(cfg.studyCells < 1 | cfg.studyCells > 15 | ...
            cfg.studyCells ~= round(cfg.studyCells))
        error('pcbtest:StudyCell', 'Study cells must be integers from 1 to 15.');
    end
    cellMap = cfg.studyCellToChannel;
    if ~isequal(sort(cellMap(:))', 1:15)
        error('pcbtest:StudyCell', 'Study cell map must contain channels 1:15 once.');
    end
    studyDir = fullfile(cfg.outputDir, 'plots_ch');
    if ~isfolder(studyDir)
        mkdir(studyDir);
    end

    % Match the filter parameters in check_xtalk_study.m.
    [highB, highA] = butter(2, 500 / (2.5e6 / 2), 'high');
    for shapingTime = cfg.shapingTimes
        rshape = 5 * shapingTime * 1e9 / 2;
        timeLabel = sprintf('%gns', shapingTime * 1e9);
        towerLabel = sprintf('T%d', cfg.tower);
        for cell = cfg.studyCells
            injectChannel = cellMap(cell);
            [referenceT, referenceWave] = loadShaped( ...
                cfg, injectChannel, injectChannel, shapingTime);
            [referenceMax, peakIndex] = max(referenceWave);
            if referenceMax <= 0
                error('pcbtest:StudySignal', ...
                    'Injected waveform has no positive peak for cell %d.', cell);
            end
            timeNs = referenceT * 1e9;
            peakNs = timeNs(peakIndex);

            offsetFig = figure('Visible', 'off');
            hold on;
            grid on;
            xlabel('time (ns)');
            xlim([0, 80 * shapingTime * 1e9]);
            marker = plot([peakNs, peakNs] + (cell - 1) * 2 * rshape, ...
                [0, 1] / (2 * rshape), 'k--');
            firstCurve = gobjects(0);
            for readoutCell = 1:15
                readoutChannel = cellMap(readoutCell);
                if readoutChannel == injectChannel
                    continue;
                end
                [t, shaped] = loadShaped( ...
                    cfg, injectChannel, readoutChannel, shapingTime);
                filtered = filter(highB, highA, shaped);
                curve = plot(t * 1e9 + 2 * rshape * (readoutCell - 1), ...
                    filtered / referenceMax);
                if isempty(firstCurve)
                    firstCurve = curve;
                end
            end
            title(sprintf(['relative xtalk signals for cell %d in ch %d ' ...
                '/ Tsh=%g ns of %s'], ...
                cell, injectChannel, shapingTime * 1e9, towerLabel));
            legend([marker, firstCurve], ...
                {sprintf('injected signal / %g', 2 * rshape), ...
                'relative xtalk for all others curves'}, 'Location', 'best');
            exportgraphics(offsetFig, fullfile(studyDir, ...
                sprintf('%s_xtalk_curves_%s_ch%d_plot.pdf', ...
                towerLabel, timeLabel, injectChannel)), ...
                'ContentType', 'vector');
            close(offsetFig);

            overlayFig = figure('Visible', 'off');
            hold on;
            grid on;
            xlabel('time (ns)');
            injected = plot(timeNs, referenceWave / referenceMax / ...
                (2 * rshape), 'k--');
            plot([peakNs, peakNs], [0, 1] / (2 * rshape), 'k--');
            firstCurve = gobjects(0);
            for readoutChannel = 1:15
                if readoutChannel == injectChannel
                    continue;
                end
                [t, shaped] = loadShaped( ...
                    cfg, injectChannel, readoutChannel, shapingTime);
                curve = plot(t * 1e9, shaped / referenceMax);
                if isempty(firstCurve)
                    firstCurve = curve;
                end
            end
            title(sprintf(['relative xtalk signals for cell %d in ch %d ' ...
                '/ Tsh=%gns'], cell, injectChannel, shapingTime * 1e9));
            legend([injected, firstCurve], ...
                {sprintf('injected signal / %g', 2 * rshape), ...
                'relative xtalk for all others curves'}, 'Location', 'best');
            exportgraphics(overlayFig, fullfile(studyDir, ...
                sprintf('%s_xtalk_curves_%s_ch%d_overlay_plot.pdf', ...
                towerLabel, timeLabel, injectChannel)), ...
                'ContentType', 'vector');
            close(overlayFig);
        end
    end
end

function [t, shaped] = loadShaped(cfg, injectChannel, readoutChannel, shapingTime)
    fileName = sprintf('%s%02d-%02d.mat', ...
        cfg.filePrefix, injectChannel, readoutChannel);
    [t, data] = pcbtest.load_waveform(fullfile(cfg.dataDir, fileName));
    nBaseline = max(1, round(numel(data) * cfg.baselineFraction));
    corrected = data - mean(data(1:nBaseline));
    shaped = pcbtest.shape_waveform( ...
        corrected, t(2) - t(1), shapingTime);
end
