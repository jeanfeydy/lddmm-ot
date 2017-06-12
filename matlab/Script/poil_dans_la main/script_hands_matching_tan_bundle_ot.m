% Matching of  multi-shapes : hands and fibres bundle (http://visionair.ge.imati.cnr.it/ontologies/shapes/)
% Author : B. Charlier (2017)

PathToFshapesTk = '../..'
addpath(genpath([PathToFshapesTk,'/Bin']))
addpath(genpath([PathToFshapesTk,'/Script']))


%------------------%
%   oliver's hand  %
%------------------%

[p,t] = read_off('./data/764-Olivier_hand_-_Simplified_to_10kF/764_closed.off');

p=p';t=t';

l =0.7;noise=.2; n=50;nb_point_bundle=8;
X0=repmat([0;-.2;0],1,n)+ randn(3,n)*.01; curvature=0.5/l; v=[1;.1;2];
Xall=generate_fiber(X0,l,curvature,nb_point_bundle,v,noise);

oliver_hand{1,1} = struct('x',p,'G',t,'f',zeros(size(p,1),1));
curvex = [];
curveG = [];
pointx = [];
for i = 1:size(Xall,3)
   curvex =[curvex ; Xall(:,:,i)'] ;
   n = size(Xall,2);
   nstart = ((i-1)*n);
   curveG = [curveG;[nstart+1:(nstart+n-1);(nstart+2):(nstart+n)]'];
   pointx =[pointx ; Xall(:,end,i)'];
end

oliver_hand{1,2} = struct('x',curvex,'G',curveG,'f',zeros(numel(Xall)/3,1)  );

%-----------------------%
%     Pierre's hand     %
%-----------------------%


[pp,tp] = read_off('./data/736-Pierre_s_hand__decimated_version/736.off');

pp=pp';
pp(:,1) = - pp(:,1);
tp=tp';


l =0.5;noise=.1; n=53;nb_point_bundle=9;
X0=repmat([0;-.2;0],1,n)+ randn(3,n)*.01; curvature=0.5/l; v=[-.1;.2;1];
Xall=generate_fiber(X0,l,curvature,nb_point_bundle,v,noise);

pierre_hand{1,1} = struct('x',pp,'G',tp,'f',zeros(size(pp,1),1));
curvex = [];
curveG = [];
pointx = [];
for i = 1:size(Xall,3)
   curvex =[curvex ; Xall(:,:,i)'] ;
   n = size(Xall,2);
   nstart = ((i-1)*n);
   curveG = [curveG;[nstart+1:(nstart+n-1);(nstart+2):(nstart+n)]'];
   pointx =[pointx ; Xall(:,end,i)'];
end

pierre_hand{1,2} = struct('x',curvex,'G',curveG,'f',zeros(numel(Xall)/3,1) );

%------------------------------%
%          parameters          %
%------------------------------%

comp_method = 'cuda';% possible values are 'cuda' or 'matlab'

% Parameters for the deformations
defo.kernel_size_mom = [.5, .2,.1]; % the kernel used to generate the deformations is a sum of 2 kernels
defo.nb_euler_steps =15; % nbr of steps in the (for||back)ward integration
defo.method =comp_method; % possible values are 'cuda' or 'matlab'

% Parameters for the matchterm
objfun{1}.normalize_objfun=1;
% First part of the shape : hand's shape surface
objfun{1}.weight_coef_dist = 6000; % weighting coeff in front of the data attachment term 
objfun{1}.distance = 'wasserstein';
objfun{1}.wasserstein_distance.epsilon = .5*(.05)^2;
objfun{1}.wasserstein_distance.niter = 200;
objfun{1}.wasserstein_distance.tau = 0; % basic sinkhorn, no extrapolation
objfun{1}.wasserstein_distance.rho = 10; % balanced case
objfun{1}.wasserstein_distance.weight_cost_varifold = [6,1]; % weight on spatial and orientation distances
objfun{1}.wasserstein_distance.method=comp_method; % possible values are 'cuda' or 'matlab'

% second part of the shape : the fibres bundles
objfun{2}.normalize_objfun=1;
objfun{2}.weight_coef_dist = 10; % weighting coeff in front of the data attachment term 
objfun{2}.distance = 'wasserstein';
objfun{2}.wasserstein_distance.epsilon = .5*(.05)^2;
objfun{2}.wasserstein_distance.niter = 200;
objfun{2}.wasserstein_distance.tau = 0; % basic sinkhorn, no extrapolation
objfun{2}.wasserstein_distance.rho = 10; % balanced case
objfun{2}.wasserstein_distance.weight_cost_varifold = [6,1]; % weight on spatial and orientation distances
objfun{2}.wasserstein_distance.method=comp_method; % possible values are 'cuda' or 'matlab'


optim.method = 'bfgs';
optim.bfgs.maxit = 300;

%----------%
%  output  %
%----------%

[momentums,summary]=fsmatch_tan(pierre_hand,oliver_hand,defo,objfun,optim);

% export in vtk files
saveDir ='./results/matching_hands_and_bundles_ot/';
export_matching_tan(pierre_hand,momentums,{zeros(size(pierre_hand{1}.f)), zeros(size(pierre_hand{2}.f))},oliver_hand,summary,saveDir)



