function [momentums,summary]=jnfmatch_geom(source,momentumsInit,defo,objfun,optim)
% Jackknife Match shape. It computes a matching between shapes using a bfgs
% optimization scheme. 
%
% Inputs:
%   templateInit:  is a cell array of fshapes containing the initial mean template.
%   momentumsInit : a cell array with momentums
%   funresInit : a cell array with functional residuals.
%   defo: is a structure containing the parameters of the deformations.
%   objfun: is a structure containing the parameters of attachment term.
%   optim: is a structure containing the parameters of the optimization procedure (here gradient descent with adaptative step)
%
% Outputs:
%   meantemplate : a cell array of fshapes containing the mean template
%   momentums: a cell array with momentums
%   funres: a cell array with functional residuals.
%   summary: is a structure containing various informations about the
%       gradient descent.
%
% See also : enr_tan_free, denr_tan_free, fsatlas_tan_free
% Author : B. Charlier (2017)


global data templateG objfunc defoc deflag templatexc templatefc


%------%
% Init %
%------%

% check data
if isempty(data)
    error('Check that data is global variable.')
end
nb_obs = size(data,1); %number of observations
nb_match = size(data,2); % number of shape in each observation

% check template
if ~iscell(source)
    source={source};
end
n = sum(cellfun(@(y) size(y.x,1),source));  %total number of points
deflag = [0,cumsum(cellfun(@(y) size(y.x,1),source))];
d = size(source{1}.x,2); % dimension of the ambient space


templateG  = cellfun(@(y) y.G,source,'uniformOutput',0);
templatexc = cellfun(@(y) y.x,source,'uniformOutput',0);
templatefc = cellfun(@(y) y.f,source,'uniformOutput',0);

% print some informations
disp_info_dimension(data,source);


% check momentums
if isempty(momentumsInit)
    momentums = zeros(d*n,1);
end

%-----------%
%  options  %
%-----------%

list_of_variables = {'p'};


% check functional residuals
if ~iscell(objfun)
    objfun ={objfun};
    objfun = objfun(ones(1,size(data,2)));
end
objfun = set_objfun_option(objfun,source,data,list_of_variables);
% compute normalization coefficients
objfun =  compute_coefficents_normalization(objfun,source,data);

defoc = set_defo_option(defo,list_of_variables);

optim = set_optim_option(optim,objfun,list_of_variables,'enr_geom');

%-----------%
% Call bfgs %
%-----------%

objfunc = objfun;

tstart = tic;

[momentums,summary] = perform_bfgs(optim.bfgs.enr_name,momentums,optim.bfgs);

momentums = reshape(momentums,n,d);

% disp informations
telapsed = toc(tstart);
fprintf('\nTotal number of iterations : %d in %g sec  (%g sec per it)\n\n',summary.bfgs.nb_of_iterations,telapsed,telapsed/summary.bfgs.nb_of_iterations);


% Summary information : save real parameters
summary.parameters.defo = defoc;
summary.parameters.objfun = objfun;
summary.parameters.optim = optim;

end

