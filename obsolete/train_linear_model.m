% train discrete net components

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
    imagepath = fullfile(datapath, label);
    mkdir(imagepath);
end

% setup caffe
addpath('/home/yz/caffe3/matlab');
modelpath = 'model/linear';
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(0);

% load caffe model
model_def = fullfile(modelpath, 'train.prototxt');
net = caffe.Net(model_def, 'test');

% load caffe solver
solver_def = fullfile(modelpath, 'solver.prototxt');
solver = caffe.Solver(solver_def);

% grab data
[ img_batch, spectra_batch ] = ...
    sampleDiscreteSpectra(T, 64);

% load data
solver.net.blobs('data').set_data(img_batch);
solver.net.blobs('label').set_data(spectra_batch);

% train
clc
% solver.solve();
iter = 6e3;
losses = zeros(1, iter);
for i=1:iter
    % grab data
    [ img_batch, spectra_batch ] = ...
        sampleDiscreteSpectra(T, 64);
    % load data
    solver.net.blobs('data').set_data(img_batch);
    solver.net.blobs('label').set_data(spectra_batch);
    % step 1
    solver.step(1);
    % get result
    losses(i) = solver.net.blobs('loss').get_data;
end

% save net
solver.net.save('linear.caffemodel');
%%
% plot prediction
solver.net.blobs('data').set_data(img_batch);
solver.net.blobs('label').set_data(spectra_batch);
output_spectra = solver.net.blobs('fc').get_data();
input_spectra = solver.net.blobs('label').get_data();
wavelength = 1:size(input_spectra, 1);
plot(wavelength, squeeze(input_spectra(:, 1)),...
    wavelength, squeeze(output_spectra(:, 1)));





