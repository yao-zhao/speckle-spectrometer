classdef DataLoader < handle
    %DATALOADER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        T % transimission matrix, first dimension is numwavelength, next dimension is numpixels
        name % file label
        savepath = 'result'% path for saving result
    end

    properties
        % general parameter
        spectra_option
        noise_option
        batchsize = 256 % batchsize
        % noise parameter
        gaussian_noise = 0.0001 % gaussian noise
        shot_noise = 0 % shot noise
        % multi splectra parameter
        numlines = 3 % number of lines
    end
    
    properties (Constant)
        spectra_options = {'single', 'multi', 'continuous'}
        noise_options = {'gaussian', 'shot', 'normal'}
    end
    
    methods (Access = public)
        % constructor
        function obj = DataLoader(filename)
            [~, obj.name, ~]= fileparts(filename);
            load_data = load(filename);
            obj.T = load_data.T;
            obj.savepath = fullfile(savepath, obj.name);
            obj.noise_option = obj.noise_options{1};
            obj.spectra_option = obj.spectra_options{2};
        end
        
        % get spectra
        function  [img_batch, spectra_batch] = getBatch(obj)
            switch lower(obj.spectra_option)
                case obj.spectra_options{1}
                    [ img_batch, spectra_batch ] = getSingleSpectra(obj);
                case obj.spectra_options{2}
                    [ img_batch, spectra_batch ] = getMultiSpectra(obj, obj.numlines);
                case obj.spectra_options{3}
                    error('not implemented')
%                     [ img_batch, spectra_batch ] = getContinuousSpectra(obj);
                otherwise
                    error('unrecognized spectra option');
            end
            switch lower(obj.noise_option)
                case obj.noise_options{1}
                    img_batch = obj.addGaussianNoise(img_batch);
                case obj.noise_options{2}
                    error('not implemented');
                case obj.noise_options{3}
                    error('not implemented');
                otherwise
                    error('unrecognized noise option');
            end
        end
        
        % get name
        function name = getName(obj)
            name = [];
            switch lower(obj.spectra_option)
                case obj.spectra_options{1}
                    name = [name, 'singleline'];
                case obj.spectra_options{2}
                    name = [name, 'multiline_',num2str(obj.numlines)];
                case obj.spectra_options{3}
                    error('not implemented')
%                     [ img_batch, spectra_batch ] = getContinuousSpectra(obj);
                otherwise
                    error('unrecognized spectra option');
            end
                
        end
        
        % show correlation plot
        function corr = getCorr(obj, maxspan)
            numW = size(obj.T, 1);
            numP = size(obj.T, 2);
            maxspan = min([numW-1, maxspan]);
            corr = zeros(maxspan, 1);
            m = obj.T;
            mm = mean(m, 1);
            m = m - ones(numW, 1)*mm;
            for span = 1:maxspan
                numspans = numW-span;
                ab = zeros(1, numP);
                a2 = zeros(1, numP);
                b2 = zeros(1, numP);
                for inum = 1:numspans
                    ab = ab + m(inum, :).*m(inum+span,:);
                    a2 = a2 + m(inum, :).^2;
                    b2 = b2 + m(inum+span,:).^2;
                end
                corr(span) = mean(ab./sqrt(a2)./sqrt(b2));
            end
        end
        
    end
    
    methods (Access = protected)
        
        % get samples multiline
        [ img_batch, spectra_batch ] = getMultiSpectra(obj, numlines);
        
        % get samples single
        [ img_batch, spectra_batch ] = getSingleSpectra(obj);
        
        % get continous
        [ img_batch, spectra_batch ] = getContinuousSpectra(obj);
        
    end
    
end

