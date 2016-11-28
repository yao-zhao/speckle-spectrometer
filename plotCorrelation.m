% plot all correlation

% train discrete net components
clear all
close all;
% generate simulated spectra and intensity pair
datapath = 'data';
files = dir(fullfile(datapath, '*.mat'));
filenames = {files.name};

% get one transimission matrix
legends = [];
for ifile = 1:length(filenames)
    filename = fullfile(datapath, filenames{ifile});
    [~, label, ~]= fileparts(filenames{ifile});
    load_data = load(filename);
    T = load_data.T;
    display(['filename', filename]);
    display(['number of pixels ', num2str(size(T,1))])
    display(['number of samplings ', num2str(size(T,2))])
    dl = DataLoader(filename);
    corr = dl.getCorr(20);
    plot(1:length(corr), corr);
    legends = [legends, {strrep(filename, '_', ' ')}];
    hold on;
end
xlabel('sampling')
ylabel('correlation')
legend(legends)
print('results/correlations.png', '-dpng')
%%

