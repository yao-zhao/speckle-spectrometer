classdef OptModel < handle
% optmization model
    
    properties
        T
        numPix
        numSpec
        method
    end
    
    properties (Constant)
        method_options = {'lsq', 'lsq_L1', 'lsq_L2'}
    end
    
    methods
        % constructor
        function obj = OptModel(T)
            obj.T = T;
            obj.numPix = size(T, 1);
            obj.numSpec = size(T, 2);
            obj.method = obj.method_options{3};
        end
        
        % fit spectra
        sp = fitSpectra(obj, I)
        
        % inference
        function [spectra, time] = inference(obj, I)
            tic;
            numbatch = size(I,4);
            spectra = zeros(obj.numSpec, numbatch);
            for ibatch = 1:numbatch
                spectra(:, ibatch) = obj.fitSpectra(I(1, :, 1, ibatch));
            end
            time = toc;
        end
        
    end

    methods (Access = protected)
        [fmin, grad] = lsq(obj, S, I)
        [fmin, grad] = lsq_L1(obj, S, I)
        [fmin, grad] = lsq_L2(obj, S, I)
    end
    
end

