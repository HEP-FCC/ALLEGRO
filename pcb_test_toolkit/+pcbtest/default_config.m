function cfg = default_config(tower)
%DEFAULT_CONFIG Settings shared by acquisition and offline analysis.
    arguments
        tower (1,1) double {mustBeInteger,mustBeNonnegative}
    end

    cfg.tower = tower;
    cfg.channels = 1:15;
    cfg.dataDir = '';
    cfg.outputDir = '';
    cfg.filePrefix = '';
    cfg.shapingTimes = [20e-9, 50e-9];
    cfg.baselineFraction = 0.10;
    cfg.displayOmitPosition = 8;
    cfg.xtalkColorLimits = [20e-9, 6e-3; 50e-9, 2e-3; 200e-9, 5e-4];
    cfg.studyCells = [3, 8];

    opticalTowers = [2, 3, 5, 7, 9, 11, 12, 14];
    logicalTowers = [4, 6, 8, 10, 13];
    if tower == 0
        cfg.displayOrder = 1:15; % T0 is the capacity reference.
        cfg.studyCellToChannel = 1:15;
    elseif ismember(tower, opticalTowers)
        cfg.displayOrder = [9, 13, 2, 5, 11, 14, 3, 8, 6, 12, 4, 10, 7, 1, 15];
        cfg.studyCellToChannel = [9, 13, 2, 5, 11, 14, 3, 6, 12, 4, 10, 7, 1, 15, 8];
    elseif ismember(tower, logicalTowers)
        cfg.displayOrder = 1:15;
        cfg.studyCellToChannel = [1:7, 9:15, 8];
    else
        error('pcbtest:UnknownTower', ...
            'No display mapping for tower %d. Add it to default_config.m.', tower);
    end

    % Apply the historical three readout presets to scope CH2.
    cfg.scope.signal = struct('position', -4, 'range', 5, 'offset', -1.8);
    cfg.scope.adjacent = struct('position', 0, 'range', 0.15, 'offset', 0.009);
    cfg.scope.other = struct('position', 0, 'range', 0.08, 'offset', 0.009);
    cfg.scope.referenceChannel = 1; % Direct generator signal; left unchanged.
    cfg.scope.readoutChannel = 2;   % Multiplexer readout waveform.
    cfg.scope.readDelaySeconds = 1;

    cfg.hardware.matrixResource = '';
    cfg.hardware.scopeResource = '';
    cfg.hardware.matrixIdContains = '';
    cfg.hardware.scopeIdContains = '';
    cfg.hardware.settleSeconds = 0;
    cfg.hardware.skipExistingFiles = true;

    % Capacity settings reproduce the legacy RC fit and C = delta(tau)/R.
    cfg.capacity.source = 'waveforms';
    cfg.capacity.resistanceOhm = 9150;
    cfg.capacity.thresholdV = 0.04;
    cfg.capacity.initialAmplitudeV = 5;
    cfg.capacity.initialTauSeconds = 1e-4;
    cfg.capacity.referenceDir = '';
    cfg.capacity.referencePrefix = '';
    cfg.capacity.referenceTauSeconds = [];
    cfg.capacity.tauDir = '';
    cfg.capacity.tauPattern = 'tau_T%d_PCBCERN.mat';
    cfg.capacity.scope = struct('position', -4, 'range', 2, 'offset', 0);
    cfg.capacity.settleSeconds = 2;
    cfg.capacity.fitComponents = true;
    cfg.capacity.lineLengthMm = [545.7, 498.9, 450.6, 451.0, ...
        452.1, 453.2, 399.7, 347.9, 294.1, 239.2, 182.5, ...
        123.8, 62.5, 4.4] + 4;
    if ismember(tower, opticalTowers)
        cfg.capacity.couplingCount = [0, 1, 0, 0, 0, 0, 6:13];
    else
        cfg.capacity.couplingCount = [0, 1, 2, 0, 0, 0, 6:13];
    end
    cfg.capacity.excludeFitPositions = [];
    cfg.capacity.plotYLimitPf = [0, 250];
    cfg.capacity.fitPlotYLimitPf = [0, 270];
    if tower == 14
        cfg.capacity.excludeFitPositions = 4;
    end
end
