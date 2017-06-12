function [u,grad_x] = dsinkhorn_log_cuda(mu,nu,x,y,epsilon,options,d)
% A wrapper function calling a cuda implementation of sinkhorn_log - stabilized 
% sinkhorn over log domain with acceleration. It computes the derivative
% with respect to x.
%
% See : Sinkhorn_log and dfshapes_wasserstein_distance functions for details
% Author : B. Charlier (2017)


options.null = 0;
rho = getoptions(options, 'rho', Inf);

if rho==Inf
    %balanced
	lambda=1;
else
    % unbalanced
    lambda = rho/(rho+epsilon);
end

% Sinkhorn
u = zeros(size(mu)); 
v = zeros(size(nu));

[u,grad_x] = dsinkhornGpuConv(u',x,y,v',epsilon,lambda,mu',nu',options.weight_cost_varifold,int32(options.niter));

u = u'; grad_x = grad_x';

end
