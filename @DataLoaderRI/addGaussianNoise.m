function [ imgbatch ] = addGaussianNoise(obj, imgbatch )
% add gaussian noise to the image batch
% gaussian noise is normalized to mean value of transmission matrix

imgbatch = imgbatch + randn(size(imgbatch)) * obj.gaussian_noise;

imgbatch(imgbatch < 0) = 0;

end

