function r=radial_function_geom(x,derivative_order,objfun)
% r=RHO(x,opt,sig) implements the Gaussian kernel and its derivatives :
%
%       rho(t)=exp(-t/lam^2);
%
% Computation of rho(x^2/2) (ie opt==0), rho'(x^2/2) (ie opt==1) and
% rho"(x^2/2)  (ie opt==2)
%
% Input :
%   x : a matrix
%   opt : a integer
%   sig : vector with kernel bandwidths
%
% Output :
%   r : matrix of the size of x
% Author : B. Charlier (2017)

r=zeros(size(x));



switch lower(objfun.kernel_geom)
    case 'gaussian'
        
        for l=objfun.kernel_size_geom
            if derivative_order==0
                r=r + exp(-x/l^2);
            elseif derivative_order==1
                r=r -exp(-x/l^2)/l^2;
            end
        end
        
    case 'cauchy'
        
        for l=objfun.kernel_size_geom
            if derivative_order==0
                r=r + 1 ./ (1 + (x/l^2));
            elseif derivative_order==1
                r=r -1 ./ (l^2 * (1 + (x/l^2)) .^2);
            end
        end
        
        
end

end
