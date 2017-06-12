function [momentums,summary]=match_geom(source,target,defo,objfun,optim)
% [momentums,summary]=MATCH_GEOM(source,target,defo,objfun,optim) computes
% a geometric matching of the shape source onto the shape target.
%
% Inputs:
%   source: a structure containing the source shape.
%   target : a structure containing the target shape.
%   defo: is a structure containing the parameters of the deformations.
%   objfun: is a structure containing the parameters of attachment term.
%   optim: is a structure containing the parameters of the optimization procedure (here gradient descent with adaptative step)
%
% Outputs:
%   momentums: a cell array with momentums
%   summary: is a structure containing various informations about the
%       gradient descent.
%
% See also : enr_geom, jnfmatch_geom
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

fprintf('\n Performing a pure geometric matching \n')

switch lower(optim.method)
    case 'bfgs'
        [momentums,summary]=jnfmatch_geom(source,[],defo,objfun,optim);
        
    case 'graddesc'
        
        % put the step size to 0
        optim.gradDesc.step_size_x = 0;
        optim.gradDesc.step_size_f = 0;
        optim.gradDesc.step_size_fr = 0;

        [~,momentums,~,summary]=jnfmean_tan_free(source,[],[],defo,objfun,optim);
            
end

end

