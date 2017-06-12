function res=scalarProductRkhsV(p1,x,defo,p2)
%  res=SCALARPRODUCTRKHSV(p1,x,defo,p2)  implements the scalar 
% product (p,K(x,x)p)/2  where the kernel size is given by 
% defo.kernel_size_mom. Several method are implemented (cuda, matlab...)
%
% scalarProductRkhsV(p1,x,defo) is equivalent to scalarProductRkhsV(p1,x,defo,p1)
% that is returns the norm squared of p1.
%
% Inputs :
%   p1 : is a n x d matrix containing initial momenta
%   x : is a n x d matrix containing positions
%   defo : structure containing the parameters of deformations
%   p2 : is a n x d matrix containing initial momenta
%
% Output :
%   res : is a real number (scalar product)
%
% See also : dnormRkhsV
% Author : B. Charlier (2017)


if nargin == 3
    p2=p1;
end

[n,d] = size(x);

switch defo.method
    case 'cuda'

        res=0;
        for sig=defo.kernel_size_mom
            res=  res + sum(sum(GaussGpuConv(x',x',p2',sig)' .* p1));
        end

    otherwise

    % Calcul de A=exp(-|x_i -x_j|^2/(2*lam^2))
    S=zeros(n);
    for l=1:d
        S=S+(repmat(x(:,l),1,n)-repmat(x(:,l)',n,1)).^2;
    end
    res=trace(p1' * rho(S,0,defo.kernel_size_mom) * p2);
end
    
    
end
