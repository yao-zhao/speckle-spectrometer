% load all transmission matrix

datapath = 'data/group1';
files = dir(fullfile(datapath, '*.mat'));
filenames = {files.name};
for ifile = 1%:length(filenames)
    filename = fullfile(datapath, filenames{ifile});
    [~, label, ~]= fileparts(filenames{ifile});
    load_data = load(filename);
    T = load_data.T;
end
% try get batch
dl = DataLoader(filename);
[img_batch, spectra_batch] = dl.getBatch();
% try training
cm = CaffeModel('fc-121', dl);
% cm.train();

%% try correlation
corr = dl.getCorr(50);
plot(1:length(corr), corr);

%%
dl = DataLoader(filename);
dl.batchsize = 16;
[img_batch, spectra_batch] = dl.getBatch();
I = (img_batch(1,:,1,1));
Sl = spectra_batch(:, 1);
om = OptModel(dl.T);
Sp = om.fitSpectra(I);
[s, t] = om.inference(img_batch);
plot(1:length(Sl), Sl, 1:length(Sp), Sp);


%%
clear all;
close all;
clc;
sch = Scheduler();
sch.train(0)

%%
sch.validate()

%% test group2
addpath('/home/yz/caffe3/matlab');
datapath = 'data/group2';
files = dir(fullfile(datapath, '*.mat'));
filenames = {files.name};
filename = fullfile(datapath, filenames{1});
% try get batch
dl = DataLoaderRI(filename);
[img_batch, spectra_batch, ri_batch] = dl.getBatch();
%
cm = CaffeModelRI('group2/conv6_fc-334', dl);
cm.train()

