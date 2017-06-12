function [dPx,dPp] = ddHrtP_tan(x,p,Px,Pp,defo)
% [dPx,dPp] = DDhRTP(x,p,Px,Pp,defo) computes the adjoints Hamiltonian 
% system usingthe methods given in defo.method.
%
% Inputs :
%   x: is a (n x d) matrix containing the points.
%   p: is a (n x d) matrix containing the momentums.
%   Px : is a (n x d) matrix (adjoint variable of x).
%   Pp : is a (n x d) matrix  (adjoint variable of p ).
%   defo : structure containing the deformations parameters
%
% Outputs
%   dPx : (n x d) matrix containing the update of Px.
%   dPp :(n x d) matrix containing the update of Pp.
%
% See also : forward_tan, backward_tan, dHr_tan
% Author : B. Charlier (2017)

switch defo.method
    case 'cuda'
        DFTP = @dftP_cuda;
    otherwise
        DFTP = @dftP_mat;
end

[dPx,dPp] = DFTP(x,p,Px,Pp,defo);

end



function [dPx,dPp]=dftP_cuda(x,p,Px,Pp,defo)
% Basic Euler step for the tangential Hamiltonian flow \deltaX_H using cuda mex file.
%
% Inputs :
%   x: is a (n x d) matrix containing the points.
%   p: is a (n x d) matrix containing the momentums.
%   Px : is a (n x d) matrix (adjoint variable x).
%   Pp : is a (n x d) matrix  (adjoint variable  momentums ).
%   defo : structure containing the deformations parameters
%
% Outputs
%   dPx : (n x d) matrix containing the update of Px.
%   dPp :(n x d) matrix containing the update of Pp.

dPp = zeros(size(Pp));
dPx = zeros(size(Px));

for lam = defo.kernel_size_mom
    
    dPx = dPx - GaussGpuGradConv(p',x',Px',lam)'+ dxxHrP2(Pp',x',p',lam)';
    dPp = dPp - GaussGpuConv(x',x',Px',lam)'+ dpxHrP2(Pp',x',p',lam)';
    
end        

end


function [dPx,dPp]=dftP_mat(x,p,Px,Pp,defo)
%Compute the [dPx;dPp] = dft([x;p]) * [Px;Pp] exactly with matlab.
%
% Inputs :
%   x: is a (n x d) matrix containing the points.
%   p: is a (n x d) matrix containing the momentums.
%   Px : is a (n x d) matrix (adjoint variable x).
%   Pp : is a (n x d) matrix  (adjoint variable  momentums ).
%   defo : structure containing the deformations parameters
%
% Outputs
%   dPx : (n x d) matrix containing the update of Px.
%   dPp :(n x d) matrix containing the update of Pp.


lam=defo.kernel_size_mom;
[n,d]=size(x);

dPp = zeros(size(Pp));
dPx = zeros(size(Px));

% Calcul de A=exp(-|x_i -x_j|^2/(lam^2))
S=zeros(n);
for l=1:d
    S=S+(repmat(x(:,l),1,n)-repmat(x(:,l)',n,1)).^2;
end
A=rho(S,0,lam);
B=rho(S,1,lam);
C=rho(S,2,lam);

for r=1:d
    dPxr=zeros(n,1);
    dPpr=-A*Px(:,r); % [\partial_{p_i}\partial_{p_j} H_r ] * Px
    % Computation of Br
    Br=2*(repmat(x(:,r),1,n)-repmat(x(:,r)',n,1)).*B;
    
    for s=1:d
        Bs=2*(repmat(x(:,s),1,n)-repmat(x(:,s)',n,1)).*B;
        dPpr=dPpr+(Bs*p(:,r)).*Pp(:,s)-Bs*(p(:,r).*Pp(:,s));% ([Diag(\partial_{x_i}\partial_{p_i} H_r)] + [\partial_{x_i} \partial_{p_j}  H_r])* Pp
        
        dPxr=dPxr-(p(:,s).*(Br*Px(:,s))+Px(:,s).*(Br*p(:,s)));
        
        Crs=4*(repmat(x(:,r),1,n)-repmat(x(:,r)',n,1))...
            .*(repmat(x(:,s),1,n)-repmat(x(:,s)',n,1)).*C;
        if s==r
            Crs=Crs+2*B;
        end
        
        for l=1:d
            dPxr=dPxr+p(:,l).*...
                ((Crs*p(:,l)).*Pp(:,s)-Crs*(p(:,l).*Pp(:,s)));
        end
    end
    dPx(:,r)=dPxr;
    dPp(:,r)=dPpr;
end


end




