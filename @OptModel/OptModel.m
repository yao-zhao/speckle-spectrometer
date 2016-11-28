classdef OptModel < handle
% optmization model
    
    properties
        T
        method
    end
    
    properties (Constant)
        method_options = {'lsq', 'lsq_L1', 'lsq_L2'}
    end
    
    methods
        % constructor
        function obj = OptModel(T)
            obj.T = T;
            obj.method = obj.method_options{3};
        end
        
        % fit spectra
        sp = fitSpectra(obj, I)
    end

    methods (Access = protected)
        [fmin, grad] = lsq(obj, S, I)
        [fmin, grad] = lsq_L1(obj, S, I)
        [fmin, grad] = lsq_L2(obj, S, I)
    end
    
end

