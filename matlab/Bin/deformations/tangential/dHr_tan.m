function [dxHr,dpHr] = dHr_tan(x,p,defo)
% [dxHr,dpHr] = DHR(x,p,defo) computes the reduced Hamiltonian system. Several
% method are implemented to speedup the compution (matlab, cuda and C). The method used is the one
% given by defo.method
%
% Input : 
%   x : state (n x d) matrix
%   p : costate (n x d) matrix
%   defo : structure containing the fields 'method' ('matlab', 'cuda' or 'grid') and 'kernel_size_mom' (kernel size)
%
% Output
%   dxHr : gradient of Hr wrt to x at point (x,p)
%   dpHr: gradient of Hr wrt to p at point (x,p)
%
% See also : forward_tan, backward_tan, ddHrtP_tan
% Author : B. Charlier (2017)


switch defo.method
    case 'cuda'
        DHR = @dHr_tan_cuda;
    case 'grid'
        DHR = @dHr_tan_grid;
    otherwise
        DHR = @dHr_tan_mat;
end

[dxHr,dpHr] = DHR(x,p,defo);

end

function [dxHr,dpHr] = dHr_tan_mat(x,p,defo)
% Matlab version of the reduced Hamiltonian system.
% Input : 
%   x : state (n x d) matrix
%   p : costate (n x d) matrix
%   defo : structure containing the field and 'kernel_size_mom' (kernel size)
%
% Output
%   dxHr : gradient of Hr wrt to x at point (x,p)
%   dpHr: gradient of Hr wrt to p at point (x,p)


[n,d]=size(x);

% Calcul de A=exp(-|x_i -x_j|^2/(lam^2))
S=zeros(n);
for l=1:d
    S=S+(repmat(x(:,l),1,n)-repmat(x(:,l)',n,1)).^2;
end
A = rho(S,0,defo.kernel_size_mom);
B = rho(S,1,defo.kernel_size_mom);


dpHr=A*p;


dxHr=zeros(n,d);
for r=1:d 
    % Computation of B=2*|x_i -x_j|*exp(-|x_i -x_j|^2/(lam^2))/(lam^2)
    Br=2*(repmat(x(:,r),1,n)-repmat(x(:,r)',n,1)).*B;
    dxHr(:,r)=dxHr(:,r)+ sum(p .* (Br*p) ,2);
end

end

function [dxHr,dpHr] = dHr_tan_cuda(x,p,defo)
% cuda version of the reduced Hamiltonian system.
% Input : 
%   x : state (n x d) matrix
%   p : costate (n x d) matrix
%   defo : structure containing the field 'kernel_size_mom' (kernel size)
%
% Output
%   dxHr : gradient of Hr wrt to x at point (x,p)
%   dpHr: gradient of Hr wrt to p at point (x,p)


dxHr = zeros(size(x));
dpHr = zeros(size(p));

if isfield(defo,'precision') && strcmp(defo.precision,'double')
    conv = @GaussGpuConvDouble;
    gradconv= @GaussGpuGrad1ConvDouble;
else
    conv = @GaussGpuConv;
    gradconv= @GaussGpuGrad1Conv;

end

for sig = defo.kernel_size_mom
    dxHr=  dxHr + gradconv(p',x',x',p',sig)';
    dpHr = dpHr + conv(x',x',p',sig)'; 
end

end
