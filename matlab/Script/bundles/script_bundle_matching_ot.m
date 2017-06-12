% Matching of  bundles fibres
% Author : B. Charlier (2017)

clear
restoredefaultpath

addpath(genpath('../../Bin'))
addpath(genpath('./data'))

target = import_fshape_vtk(['./data/bundle_target.vtk']);
source = import_fshape_vtk(['./data/bundle_source.vtk']);
source.f = repmat( linspace(0,1,10)', size(source.x,1) /10 ,1 )

%------------------------------%
%          parameters          %
%------------------------------%

comp_method = 'matlab';% possible values are 'cuda' or 'matlab'

% Parameters for the deformations
defo.kernel_size_mom = [.4]; % the kernel used to generate the deformations is a sum of 2 kernels
defo.nb_euler_steps =15; % nbr of steps in the (for||back)ward integration
defo.method =comp_method; % possible values are 'cuda' or 'matlab'

% Parameter for the optimization
optim.method = 'bfgs';
optim.bfgs.maxit = 150;

% Parameters for the matchterm
objfun{1}.weight_coef_dist = 1000; % weighting coeff in front of the data attachment term 
objfun{1}.normalize_objfun=1;

%Wasserstein
objfun{1}.distance = 'wasserstein';
objfun{1}.wasserstein_distance.epsilon = .005;
objfun{1}.wasserstein_distance.niter = 200;
objfun{1}.wasserstein_distance.tau = 0; % basic sinkhorn, no extrapolation
objfun{1}.wasserstein_distance.rho = 1; % balanced case
objfun{1}.wasserstein_distance.weight_cost_varifold = [6,1]; % weight on spatial and orientation distances
objfun{1}.wasserstein_distance.method=comp_method; % possible values are 'cuda' or 'matlab'

[momentums,summary]=match_geom(source,target,defo,objfun,optim);

saveDir =['./results/matching_bundles_wasserstein/',date];
export_matching_tan(source,momentums,zeros(size(source{1}.x,1),1),target,summary,saveDir)
