function [fmin, grad] = lsq_L2(obj, S, I)
% the target function to be optimzied
% input: 
%   S: numspec * 1
%   T: numpix * numspec
%   I: numpix * 1
% output:
%   fmin: fmin value
%   grad: grad over S

    diff = obj.T*S - I;
    fmin = sum(diff.^2) + obj.lambda * sum(S.^2);
    grad = 2 * (obj.T' * diff + obj.lambda * S);
end