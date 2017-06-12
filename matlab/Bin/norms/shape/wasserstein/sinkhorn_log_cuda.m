function [u,v,Wprimal,Wdual,err] = sinkhorn_log_cuda(mu,nu,x,y,epsilon,options)
% A wrapper function calling a cuda implementation of sinkhorn_log - stabilized sinkhorn over log domain without acceleration
%
% See : Sinkhorn_log function for details
% Author : B. Charlier (2017)


rho = getoptions(options, 'rho', Inf);

% Sinkhorn
u = zeros(size(mu)); 
v = zeros(size(nu));

if rho==Inf
    %balanced
    lambda=1;
    [u,v,Wdual] = sinkhornGpuConv(u',x,y,v',epsilon,lambda,mu',nu',options.weight_cost_varifold,int32(options.niter));
else
    % unbalanced
    lambda = rho/(rho+epsilon);
    [u,v,Wdual] = sinkhornGpuConv_unbalanced(u',x,y,v',epsilon,lambda,rho,mu',nu',options.weight_cost_varifold,int32(options.niter));
end



Wprimal = [];
err =0;

u = u'; v = v';

end


