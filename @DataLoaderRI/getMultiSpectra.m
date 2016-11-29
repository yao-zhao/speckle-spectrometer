function [ img_batch, spectra_batch, ri_batch ] = getMultiSpectra(obj, numlines)
% get a random combination of discrerete spectra
% total intensities are normalized to equal to 1
% input: 
%   T is the transmission matrix
%   batchsize is the batchsize
%   numlines is number of lines
% output:
%   img_batch are the images in batch
%   spectra_batch are the labels in batch

T = obj.T;
batchsize = obj.batchsize;
[numpix, numspec, numri] = size(T);
spectra_batch = zeros(numspec, batchsize);
img_batch = zeros(1, numpix, 1, batchsize);
ri_batch = zeros(1, batchsize);

for ibatch = 1:batchsize
    ri = randi(numri);
    index = randsample(numspec, numlines);
    spectra = zeros(numspec,1);
    spectra(index) = rand(numlines, 1);
    spectra = spectra / sum(spectra);
    img = T(:, :, ri) * spectra;
    spectra_batch(:, ibatch) = spectra;
    img_batch(1, :, 1, ibatch) = img;
    ri_batch(1, ibatch) = ri-1;
end

end

