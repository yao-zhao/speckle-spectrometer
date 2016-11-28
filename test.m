% load all transmission matrix
clear all
close all;
datapath = 'data';
files = dir(fullfile(datapath, '*.mat'));
filenames = {files.name};
for ifile = 1%:length(filenames)
    filename = fullfile(datapath, filenames{ifile});
    [~, label, ~]= fileparts(filenames{ifile});
    load_data = load(filename);
    T = load_data.T;
end
%% try get batch
dl = DataLoader(filename);
[img_batch, spectra_batch] = dl.getBatch();
%% try training
cm = CaffeModel('fc-121', dl);
cm.train();

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



