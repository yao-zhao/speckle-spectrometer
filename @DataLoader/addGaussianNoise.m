function [ imgbatch ] = addGaussianNoise(obj, imgbatch )
% add gaussian noise to the image batch

imgbatch = imgbatch + randn(size(imgbatch)) * obj.gaussian_noise;

imgbatch(imgbatch < 0) = 0;

end

