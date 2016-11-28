% train discrete net components
clear all
close all;
% generate simulated spectra and intensity pair
datapath = 'data';
files = dir(fullfile(datapath, '*.mat'));
filenames = {files.name};

% get one transimission matrix
for ifile = 1%:length(filenames)
    filename = fullfile(datapath, filenames{ifile});
    [~, label, ~]= fileparts(filenames{ifile});
    load_data = load(filename);
    T = load_data.T;
end

dl = DataLoader(filename);
[img_batch, spectra_batch] = dl.getBatch();

cm = CaffeModel('cnn-multiple', dl);
cm.train();