function [ S ] = fitSpectra(obj, I )
% fit the spectra given a img
% fmin = \sum{s * T - img)^2} + \sum{s^2}

[numPix, numSpec] = size(obj.T);

assert(numel(I) == numPix, 'check image dimension failed');
I = reshape(I, numPix, 1);
ini = zeros(numSpec, 1) + 1/numSpec;
lb = zeros(numSpec, 1);
ub = ones(numSpec, 1);
options = optimoptions('fmincon','Display','off',...
    'Algorithm', 'trust-region-reflective',...
    'SpecifyObjectiveGradient',true);

switch obj.method
    case obj.method_options{1}
        error('not implemented')
    case obj.method_options{2}
        error('not implemented')
    case obj.method_options{3}
        targetfunc = @(S)obj.lsq_L2(S, I);
    otherwise
        error('unknown method')
end


S = fmincon(targetfunc, ini, [], [], [], [], lb, ub, [], options);


end

