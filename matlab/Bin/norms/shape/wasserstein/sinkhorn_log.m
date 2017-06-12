function [u,v,gamma,Wprimal,Wdual,err] = sinkhorn_log(mu,nu,c,epsilon,options)
% sinkhorn_log - stabilized sinkhorn over log domain with acceleration
%
%   [u,v,Wprimal,Wdual,err] = sinkhorn_log(mu,nu,c,epsilon,options);
%
%   mu and nu are marginals.
%   c is cost
%   epsilon is regularization
%   coupling is 
%       gamma = exp( (-c+u*ones(1,N(2))+ones(N(1),1)*v')/epsilon );
%
%   options.niter is the number of iterations.
%   options.tau is an avering step. 
%       - tau=0 is usual sinkhorn
%       - tau<0 produces extrapolation and can usually accelerate.
%
%   options.rho controls the amount of mass variation. Large value of rho
%   impose strong constraint on mass conservation. rho=Inf (default)
%   corresponds to the usual OT (balanced setting). 
% Author : B. Charlier (2017)


options.null = 0;
niter = getoptions(options, 'niter', 1000);
tau  = getoptions(options, 'tau', -.5);
rho = getoptions(options, 'rho', Inf);
record_evol = getoptions(options, 'record_evol', 0);


lambda = rho/(rho+epsilon);
if rho==Inf
    lambda=1;
end

% Sinkhorn
N = [size(mu,1) size(nu,1)];
ave = @(tau, u,u1)tau*u+(1-tau)*u1;
lse = @(A)logReg(sum(exp(A),2));
M = @(u,v)(-c+ repmat(u,1,N(2)) +  repmat(v',N(1), 1) )/epsilon;

if record_evol %only if needed!
    err = zeros(niter,1);
    Wprimal = zeros(niter,1);
    Wdual = zeros(niter,1);
end

u = zeros(N(1),1); 
v = zeros(N(2),1);


for i=1:niter
    
    u1 = u;
    u = ave(tau, u, ...
	lambda*epsilon*log(mu) - lambda*epsilon*lse( M(u,v) ) + lambda*u );

    v = ave(tau, v, ...
	lambda*epsilon*log(nu) - lambda*epsilon*lse( M(u,v)' ) + lambda*v );

    if record_evol %only if needed!
        % coupling
        gamma = exp(M(u,v));        
        [Wprimal(i),Wdual(i),err(i)]= objfunction(u,u1,v,nu,mu,c,gamma,rho,epsilon);
    end
    
end
   
% prepare output

if ~record_evol
    % coupling
    gamma = exp(M(u,v));    
    [Wprimal,Wdual,err]= objfunction(u,u1,v,nu,mu,c,gamma,rho,epsilon);
    
end

end



function [Wprimal,Wdual,err]= objfunction(u,u1,v,nu,mu,c,gamma,rho,epsilon)

% kullback divergence
H = @(p)-sum( p(:).*(logReg(p(:))-1) );
KL  = @(h,p)sum( h(:).*log( h(:)./p(:) ) - h(:)+p(:) );
KLd = @(u,p)sum( p(:).*(exp(-u(:))-1) );
dotp = @(x,y)sum(x(:).*y(:));


if rho==Inf % marginal violation
    
    Wprimal = dotp(c,gamma) - epsilon*H(gamma);
    Wdual = dotp(u,mu) + dotp(v,nu) ...
        - epsilon*sum( gamma(:) );
    err = norm( sum(gamma,2)-mu );
    
else % difference with previous iterate
    
    Wprimal = dotp(c,gamma) - epsilon*H(gamma) ...
        + rho*KL(sum(gamma,2),mu) ...
        + rho*KL(sum(gamma,1),nu);
    Wdual = -rho*KLd(u/rho,mu) - rho*KLd(v/rho,nu) ...
        - epsilon*sum( gamma(:) );
    err = norm(u(:)-u1(:), 1);
    
end

end

function r = logReg(x)
 r = log(x+eps);
end
