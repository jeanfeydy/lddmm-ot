function [momentums,summary]=fsmatch_tan(source,target,defo,objfun,optim)
% [momentums,funres,List_energy]=FSMATCH_TAN(source,target,defo,objfun,optim)
% performs a geometrico-functional  matching of the fshape "source" to the 
% fshape "target" in the tangential framework.
%
% Note: This function is a wrapper function as it simply call the low-level function jnfmeanMultiShape with
% some particular parameters.
%
% Inputs:
%   source: is a structure containing the source fshape
%   target: is a structure containing the target fshape
%   defo: is a structure containing the parameters of the deformations.
%   objfun: is a structure containing the parameters of attachment term.
%   optim: is a structure containing the parameters of the optimization procedure
%
% Outputs:
%   momentums: is a matrix containing the momentums (geometric deformation).
%   funres: is a vector with the functional residuals (functional deformation).
%   List_energy : is a matrix containing the values of the energy during
%       optimization process.
%
% See also: jnfmean_tan_free,fsatlas_tan_free,fsatlas_tan_HT
% Author : B. Charlier (2017)


global data

%--------%
%  DATA  %
%--------%


if ~iscell(target)
    data= {target};
else
    data = target;
end

[nb_obs,nb_match] = size(data);

if nb_obs >1
	error('target must contain only 1 observation')
end


for i = 1:nb_obs
    for l=1:nb_match
        if ~isfield(data{i,l},'f')
            data{i,l}.f = zeros(size(data{i,l}.x,1),1); 
        end
    end
end

%----------%
% TEMPLATE %
%----------%

if ~iscell(source)
    source= {source};
end

for l=1:nb_match
    if ~isfield(source{l},'f')
        source{l}.f = zeros(size(source{l}.x,1),1);
    end
end



%--------%
%  MATCH %
%--------%

switch lower(optim.method)
    case 'bfgs'
        fprintf('\n Performing a geometrico-functional matching \n')
        [momentums,summary]=jnfmatch_tan(source,[],[],defo,objfun,optim);

end

end






