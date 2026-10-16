% Analyze complete cross-talk matrices for the selected towers.
toolkitDir = fileparts(mfilename('fullpath'));
addpath(toolkitDir);

towers = [4];                  % For multiple towers, use e.g. [3, 4, 6].
dataDir = fullfile(toolkitDir, 'data', 'xtalk_run_001');
datasetLabel = 'PCBCHINA';
outputLabel = 'PCBCHINA';       % Results: output/<outputLabel>/xtalk/.
towerSubfolders = true;         % Windows acquisition saves each tower under T<number>.
shapingTimes = [20e-9, 50e-9, 200e-9];
studyCells = [3, 8];           % Set [] to skip the two waveform study views.

for tower = towers
    cfg = pcbtest.default_config(tower);
    cfg.dataDir = dataDir;
    if towerSubfolders
        cfg.dataDir = fullfile(cfg.dataDir, sprintf('T%d', tower));
    end
    cfg.filePrefix = sprintf('T%d_xtalk_%s_', tower, datasetLabel);
    cfg.outputDir = fullfile(toolkitDir, 'output', outputLabel, 'xtalk', ...
        sprintf('T%d', tower));
    cfg.shapingTimes = shapingTimes;
    cfg.studyCells = studyCells;
    pcbtest.analyze_xtalk(cfg);
    pcbtest.plot_xtalk_study(cfg);
end
