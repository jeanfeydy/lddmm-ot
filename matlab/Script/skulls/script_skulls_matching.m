% Matching of two curves : Skulls dataset
% Author : B. Charlier (2017)

clear all
restoredefaultpath
addpath(genpath('../../Bin'))

%----------------%
%      Data      %
%----------------%

%choose a target
hom = 'australopithecus';
hom = 'sapiens';
hom = 'erectus';

r =1; %the code should be scale invariant...

target = import_fshape_vtk(['./Data/skull_',hom,'.vtk']); 
target.f = zeros(size(target.x,1),1);
nu = sum(area(target.x,target.G));
target.x = r* target.x /nu;


template = import_fshape_vtk('./Data/template.vtk')
template.f = zeros(size(template.x,1),1);
mu = sum(area(template.x,template.G));
template.x = r* template.x /mu;


%------------------------------%
%          parameters          %
%------------------------------%

comp_method = 'matlab';% possible values are 'cuda' or 'matlab'

% Parameters for the deformations
defo.kernel_size_mom = r*[.06,.03,.013]; % size of the kernel used to generate the deformations
defo.nb_euler_steps =10; % nbr of steps in the (for||back)ward integration
defo.method =comp_method; % possible values are 'cuda' or 'matlab'

% Parameters for data attachment term 
objfun.weight_coef_dist = 10000; % weighting coeff in front of the fidelity term
objfun.distance = 'wasserstein'; % OT fidelity
objfun.wasserstein_distance.method=comp_method;
objfun.wasserstein_distance.epsilon = .5*(r*.01/6)^2;
objfun.wasserstein_distance.niter = 450;
objfun.wasserstein_distance.tau = 0; % basic sinkhorn, no extrapolation (only matlab version)
objfun.wasserstein_distance.rho = Inf; % balanced case
objfun.wasserstein_distance.weight_cost_varifold = [1,0.001]; % weight on spatial and orientation distances

% Parameters for optimization
optim.method='bfgs' 
optim.bfgs.maxit=200;

[momentums,summary]=match_geom(template,target,defo,objfun,optim);

export_matching_tan(template,momentums,template.f,target,summary,['results/matching_',hom])
