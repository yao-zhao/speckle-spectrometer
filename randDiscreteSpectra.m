function [ img_batch, spectra_batch ] = randDiscreteSpectra(T, batchsize, numsample)
%get a random combination of discrerete random spectra

% sample 1
[numpix, numspec] = size(T);

spectra_batch = zeros(numspec, batchsize);
img_batch = zeros(1, numpix, 1, batchsize);

for ibatch = 1:batchsize
    index = randsample(numspec, numsample);
    spectra = zeros(numspec,1);
    spectra(index) = rand(numsample, 1);
    spectra = spectra / sum(spectra);
%     img = sum(T*spectra, 2);
    img = T*spectra;
    spectra_batch(:, ibatch) = spectra;
    img_batch(1, :, 1, ibatch) = img;
    
    %    subplot(1,2,1)
    %    plot(spectra);
    %    subplot(1,2,2);
    %    plot(img);
end

end

