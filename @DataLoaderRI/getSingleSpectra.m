function [ img_batch, spectra_batch ] = getSingleSpectra(obj)
% sample a single spectra
% total intensities are normalized to equal to 1
% input: 
%   T is the transmission matrix
%   batchsize is the batchsize
% output:
%   img_batch are the images in batch
%   spectra_batch are the labels in batch

T = obj.T;
batchsize = obj.batchsize;
[numpix, numspec] = size(T);

spectra_batch = zeros(numspec, batchsize);
img_batch = zeros(1, numpix, 1, batchsize);

index = randsample(numspec, batchsize);
for ibatch = 1:batchsize
    spectra = zeros(numspec,1);
    spectra(index(ibatch)) = 1;
    img = T*spectra;
    spectra_batch(:, ibatch) = spectra;
    img_batch(1, :, 1, ibatch) = img;
end

end

