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


addpath('/home/yz/caffe3/matlab');
modelpath = 'model/cnn';
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(0);

model_def = fullfile(modelpath, 'train.prototxt');
net = caffe.Net(model_def, 'test');

% load caffe solver
solver_def = fullfile(modelpath, 'solver.prototxt');
solver = caffe.Solver(solver_def);

% train
clc

batchsize = 256;
numsample = 5;

iter = 6e3;
losses = zeros(1, iter);
for i=1:iter
    % grab data
    [ img_batch, spectra_batch ] = ...
        randDiscreteSpectra(T, batchsize, numsample);
    % load data
    solver.net.blobs('data').set_data(img_batch);
    solver.net.blobs('label').set_data(spectra_batch);
    % step 1
    solver.step(1);
    % get result
    losses(i) = solver.net.blobs('loss').get_data;
    % plot
    if mod(i, 100) == 0
        % plot prediction
        output_spectra = solver.net.blobs('fc').get_data();
        input_spectra = solver.net.blobs('label').get_data();
        wavelength = 1:size(input_spectra, 1);
        plot(wavelength, squeeze(input_spectra(:, 1)),...
            wavelength, squeeze(output_spectra(:, 1)));
        legend({'label','fc'})
        pause(.1)
    end
end



%%
solver.net.save('cnn.caffemodel');

%%


