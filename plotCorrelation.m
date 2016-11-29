% plot all correlation

% % train discrete net components
% clear all
% close all;
% % generate simulated spectra and intensity pair
% datapath = 'data';
% files = dir(fullfile(datapath, '*.mat'));
% filenames = {files.name};
% legends = [];
% for ifile = 1:length(filenames)
%     filename = fullfile(datapath, filenames{ifile});
%     [~, label, ~]= fileparts(filenames{ifile});
%     load_data = load(filename);
%     T = load_data.T;
%     display(['filename', filename]);
%     display(['number of pixels ', num2str(size(T,1))])
%     display(['number of samplings ', num2str(size(T,2))])
%     dl = DataLoader(filename);
%     corrW = dl.getWavelengthCorr(20);
%     plot(1:length(corrW), corrW);
%     legends = [legends, {strrep(filename, '_', ' ')}];
%     hold on;
% end
% title('wavelength correlation');
% xlabel('wavelength increments')
% ylabel('correlation')
% legend(legends)
% print('results/wavelength_correlations.png', '-dpng')
% %%
% close all;
% legends = [];
% for ifile = 1:length(filenames)
%     filename = fullfile(datapath, filenames{ifile});
%     [~, label, ~]= fileparts(filenames{ifile});
%     load_data = load(filename);
%     T = load_data.T;
%     display(['filename', filename]);
%     display(['number of pixels ', num2str(size(T,1))])
%     display(['number of samplings ', num2str(size(T,2))])
%     dl = DataLoader(filename);
%     corrW = dl.getPixelCorr(50);
%     plot(1:length(corrW), corrW);
%     legends = [legends, {strrep(filename, '_', ' ')}];
%     hold on;
% end
% title('pixel correlation');
% xlabel('wavelength increments')
% ylabel('correlation')
% legend(legends)
% print('results/pixel_correlations.png', '-dpng')


%% group2
clear all
close all;
datapath = 'data/group2';
files = dir(fullfile(datapath, '*.mat'));
filenames = {files.name};
legends = [];
for ifile = 1:length(filenames)
    filename = fullfile(datapath, filenames{ifile});
    [~, label, ~]= fileparts(filenames{ifile});
    load_data = load(filename);
    T = load_data.T;
    display(['filename', filename]);
    display(['number of pixels ', num2str(size(T,1))])
    display(['number of samplings ', num2str(size(T,2))])
    dl = DataLoaderRI(filename);
    corrW = dl.getWavelengthCorr(20);
    plot(1:length(corrW), corrW);
    legends = [legends, {strrep(filename, '_', ' ')}];
    hold on;
end
title('wavelength correlation');
xlabel('wavelength increments')
ylabel('correlation')
legend(legends)
print('results/group2/wavelength_correlations.png', '-dpng')
%%
close all;
legends = [];
for ifile = 1:length(filenames)
    filename = fullfile(datapath, filenames{ifile});
    [~, label, ~]= fileparts(filenames{ifile});
    load_data = load(filename);
    T = load_data.T;
    display(['filename', filename]);
    display(['number of pixels ', num2str(size(T,1))])
    display(['number of samplings ', num2str(size(T,2))])
    dl = DataLoaderRI(filename);
    corrW = dl.getPixelCorr(50);
    plot(1:length(corrW), corrW);
    legends = [legends, {strrep(filename, '_', ' ')}];
    hold on;
end
title('pixel correlation');
xlabel('pixel increments')
ylabel('correlation')
legend(legends)
print('results/group2/pixel_correlations.png', '-dpng')

