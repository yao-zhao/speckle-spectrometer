classdef DataLoaderRI < handle    
    
    properties
        T % transimission matrix, first dimension is numwavelength, next dimension is numpixels
        name % file label
        savepath = 'result'% path for saving result
        numPix
        numSpec
        numRI
    end
    
    properties
        % general parameter
        spectra_option
        noise_option
        batchsize = 256 % batchsize
        % noise parameter
        gaussian_noise_ratio = 1e-3 % gaussian noise
        gaussian_noise
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
        function obj = DataLoaderRI(filename)
            [~, obj.name, ~]= fileparts(filename);
            load_data = load(filename);
            obj.T = load_data.T;
            obj.numPix = size(obj.T, 1);
            obj.numSpec = size(obj.T, 2);
            obj.numRI = size(obj.T, 3);
            obj.savepath = fullfile(obj.savepath, obj.name);
            obj.noise_option = obj.noise_options{1};
            obj.spectra_option = obj.spectra_options{2};
            obj.gaussian_noise = mean(obj.T(:)) * obj.gaussian_noise_ratio;
        end
        
        % get spectra
        function  [img_batch, spectra_batch, ri_batch] = getBatch(obj)
            switch lower(obj.spectra_option)
                case obj.spectra_options{1}
                    error('not implemented')
                case obj.spectra_options{2}
                    [ img_batch, spectra_batch, ri_batch ] = getMultiSpectra(obj, obj.numlines);
                case obj.spectra_options{3}
                    error('not implemented')
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
        function name = getSaveName(obj)
            name = [obj.name, '-'];
            switch lower(obj.spectra_option)
                case obj.spectra_options{1}
                    name = [name, 'singleline'];
                case obj.spectra_options{2}
                    name = [name, 'multiline_',num2str(obj.numlines)];
                case obj.spectra_options{3}
                    error('not implemented')
                    %[ img_batch, spectra_batch ] = getContinuousSpectra(obj);
                otherwise
                    error('unrecognized spectra option');
            end
            name = [name, '-'];
            switch lower(obj.noise_option)
                case obj.noise_options{1}
                    name = [name, 'gaussian_',num2str(obj.gaussian_noise_ratio)];
                case obj.noise_options{2}
                    error('not implemented')
                case obj.noise_options{3}
                    error('not implemented')
                otherwise
                    error('unrecognized noise option');
            end
            name = [name, '-'];
        end
        
%         show correlation plot
        function corr = getWavelengthCorr(obj, maxspan)
            numP = size(obj.T, 1);
            numW = size(obj.T, 2);
            maxspan = min([numP-1, maxspan]);
            corr = zeros(maxspan, 1);
            m = obj.T(:,:,1);
            mm = mean(m, 2);
            m = m - mm * ones(1, numW);
            for span = 1:maxspan
                numspans = numW-span;
                ab = zeros(numP, 1);
                a2 = zeros(numP, 1);
                b2 = zeros(numP, 1);
                for inum = 1:numspans
                    ab = ab + m(:, inum).*m(:, inum+span);
                    a2 = a2 + m(:, inum).^2;
                    b2 = b2 + m(:, inum+span).^2;
                end
                corr(span) = mean(ab./sqrt(a2)./sqrt(b2));
            end
        end
        
%         get pixel correlation plot
        function corr = getPixelCorr(obj, maxspan)
            numP = size(obj.T, 1);
            numW = size(obj.T, 2);
            maxspan = min([numP-1, maxspan]);
            corr = zeros(maxspan, 1);
            m = obj.T(:,:,1);
            mm = mean(m, 1);
            m = m - ones(numP, 1)*mm;
            for span = 1:maxspan
                numspans = numW-span;
                ab = zeros(1, numW);
                a2 = zeros(1, numW);
                b2 = zeros(1, numW);
                for inum = 1:numspans
                    ab = ab + m(inum, :).*m(inum+span, :);
                    a2 = a2 + m(inum, :).^2;
                    b2 = b2 + m(inum+span, :).^2;
                end
                corr(span) = mean(ab./sqrt(a2)./sqrt(b2));
            end
        end
        
        
    end
    
    methods (Access = protected)
        
        % get samples multiline
        [ img_batch, spectra_batch, ri_batch ] = getMultiSpectra(obj, numlines);
        
        % get samples single
%         [ img_batch, spectra_batch ] = getSingleSpectra(obj);
        
        % get continous
%         [ img_batch, spectra_batch ] = getContinuousSpectra(obj);
        
        
    end
    
end

