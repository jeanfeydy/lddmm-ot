function [dx,dp]=dnormRkhsV(x,p,defo)
%  [dx,dp]=DNORMrKHSV(x,p,defo) computes the gradient of (p,K(x,x)p)/2 wrt 
% x and p. The kernel size is given by defo.kernel_size_mom. 
% Several method are implemented (cuda,  matlab...)
%
% Inputs :
%   x : is a n x d matrix containing positions
%   p : is a n x d matrix containing momentums
%   defo : structure containing the parameters of deformations
%
% Output :
%   dx : is a (n x d) column vector
%   dp : is a (n x d) column vector
%
% See also : scalarProductRkhsV
% Author : B. Charlier (2017)


[n,d] = size(x);

switch defo.method
    case 'cuda'
        
	dx=zeros(n,d);
        dp=zeros(n,d);
        for t=size(defo.kernel_size_mom,2)
            dx = dx + GaussGpuGrad1Conv(p',x',x',p',defo.kernel_size_mom(t))';
            dp = dp + GaussGpuConv(x',x',p',defo.kernel_size_mom(t))';
        end

    otherwise
        
	% Calcul de A=exp(-|x_i -x_j|^2/(2*lam^2))
        dx=zeros(n,d);
        dp=zeros(n,d);
        S=zeros(n);
        pp=zeros(n);
        for l=1:d
            S=S+(repmat(x(:,l),1,n)-repmat(x(:,l)',n,1)).^2;
            pp=pp+p(:,l)*p(:,l)';
        end
        A=rho(S,0,defo.kernel_size_mom);
        B=2*rho(S,1,defo.kernel_size_mom).*pp;
        
        for r=1:d
            dx(:,r)=sum(B.*(repmat(x(:,r),1,n)-repmat(x(:,r)',n,1)),2);
            dp(:,r)=A*p(:,r);
        end
end
end

