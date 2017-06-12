function [momentums,summary]=jnfmatch_tan(templateInit,momentumsInit,funresInit,defo,objfun,optim)
% Jackknife Mean fshape estimation in the 'tangential' and 'free' framework.
% This is a gradient descent for cost function computed in enr_tan_free.
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
if ~iscell(templateInit)
    templateInit={templateInit};
end

deflag = [0,cumsum(cellfun(@(y) size(y.x,1),templateInit))];
n = deflag(end);
d = size(templateInit{1}.x,2); % dimension of the ambient space

templateG  = cellfun(@(y) y.G,templateInit,'uniformOutput',0);
templatexc = cellfun(@(y) y.x,templateInit,'uniformOutput',0);
templatefc = cellfun(@(y) y.f,templateInit,'uniformOutput',0);


% print some informations
disp_info_dimension(data,templateInit);

% check momentums
if isempty(momentumsInit)
    momentums = zeros(d*n,1);
else 
    momentums = momentumsInit;
end


%-----------%
%  options  %
%-----------%

list_of_variables = {'p','fr'};

if ~iscell(objfun)
    objfun ={objfun};
    objfun = objfun(ones(1,size(data,2)));
end
objfun = set_objfun_option(objfun,templateInit,data,list_of_variables);
% compute normalization coefficients
objfun =  compute_coefficents_normalization(objfun,templateInit,data);

defoc = set_defo_option(defo,list_of_variables);

optim = set_optim_option(optim,objfun,list_of_variables,'enr_match_tan');


%------------------%
%       BFGS       %
%------------------%

tstart = tic;
objfunc = objfun;

[mom,summary] = perform_bfgs(optim.bfgs.enr_name,momentums,optim.bfgs);

momentums = reshape(mom(1:n*d),n,d);


% disp informations
telapsed = toc(tstart);
fprintf('\nTotal number of iterations : %d in %g sec  (%g sec per it)\n\n',summary.bfgs.nb_of_iterations,telapsed,telapsed/summary.bfgs.nb_of_iterations);


% Prepare Output
meantemplate = cell(1,nb_match);
for l=1:nb_match
    meantemplate{1,l} = struct('x',templatexc{l},'f',templatefc{l},'G',templateG{l});
end

% Summary information : save real parameters
summary.parameters.defo = defoc;
summary.parameters.objfun = objfun;
summary.parameters.optim = optim;

end

