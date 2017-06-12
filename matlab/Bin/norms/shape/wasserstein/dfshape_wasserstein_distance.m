function [dxg]= dfshape_wasserstein_distance(fs1,fs2,objfun)
% DFSHAPE_KERNEL_DISTANCE(templatefinal,target,objfun) computes the derivatives of the kernel based
% distances between fshapes.
%
%  \sum_i\sum_j K_signal(f_i-g_j)^2) K_geom(-norm(x_i-y_j)^2) K_tan(angle(V_i,W_j))
%
% Possible method are 'cuda' or 'matlab'.
%
% Inputs:
%   templatefinal : "fshape structure" containing the shooted template
%   target : "fshape structure" containing the target.
%   objfun : is a structure containing the data attachment term parameters (mandatory fields are : 'kernel_geom', 'kernel_signal' and 'kernel_grass' and the associated bandwidth)
% Output
%   dxg : a matrix of the size of templatefinal.x.
%   dfg : a matrix of the size of templatefinal.f.
% Author : B. Charlier (2017)



%------------------------%
% Start the chain's rule %
%------------------------%

d_ambient_space=size(fs1.x,2);
m=size(fs1.G,2)-1;

% discretize the fshapes
[center_faceX,normalsX]=fcatoms(fs1.x,fs1.G);
[center_faceY,normalsY]=fcatoms(fs2.x,fs2.G);

options = objfun.wasserstein_distance;


if min(m, d_ambient_space-m) ==0 % for points clouds or simplexes dim or codim == 0 : some simplifications occurs
    
    x=center_faceX';
    y=center_faceY';
    
    mu = fs1.G;
    nu = fs2.G;
    
    %only needed for matlab version. See built-in function for cuda version.
    nablaC = @(x,y)repmat(x,[1 1 size(y,2)]) - ...
        repmat(reshape(y,[size(y,1) 1 size(y,2)]),[1 size(x,2)]);
    C = @(x,y)squeeze( sum(nablaC(x,y).^2)/2 );% C(x,y)=1/2*|x-y|^2
    
elseif min(m,d_ambient_space-m) ==1 % for curves or surface dim or codim ==1;
    
    x=[center_faceX';normalsX'];
    y=[center_faceY';normalsY'];
    
    mu = area(fs1.x,fs1.G);
    nu = area(fs2.x,fs2.G);
    
    %only needed for matlab version. See built-in function for cuda version.
    C = @(X,Y) cost_varifold(X,Y,options.weight_cost_varifold);
    nablaC = @(X,Y) dcost_varifold(X,Y,options.weight_cost_varifold);
    
end


switch options.method
    case 'matlab'
        
        [u,~,gamma,~,~,~] = sinkhorn_log(mu,nu,C(x,y),options.epsilon,options);
        
        % gradient with respect to positions
        DXg = sum( nablaC(x,y) .*...
            repmat(reshape(gamma, [1,size(mu,1),size(nu,1)]), [size(x,1),1,1]),  ...
            3 );
    case 'cuda'
        
        [u,DXg] = dsinkhorn_log_cuda(mu,nu,x,y,options.epsilon,options,d_ambient_space);
        DXg = DXg';
        
end

% gradient with respect to the weights
if options.rho==Inf% balanced
    Dmug = u;
else% unbalanced
    Dmug = -options.rho*(exp(-u/options.rho)-1);
end

% gradient with respect to the normal and centerFace
if min(m,d_ambient_space-m) == 0
    
    Dcenter_faceXg = DXg(1:d_ambient_space,:)';
    DnormalsXg = zeros(size(normalsX));

    
elseif min(m,d_ambient_space-m) ==1
    
    norm_normalsX = sqrt(sum(normalsX .^2,2));
    unit_normalsX = normalsX ./  repmat(norm_normalsX,1,d_ambient_space);
    
    Dcenter_faceXg = DXg(1:d_ambient_space,:)';
    DnormalsXg = repmat(Dmug,1,d_ambient_space) .*unit_normalsX+ DXg(d_ambient_space+1:2*d_ambient_space,:)'/2;
    
    
end


%-------------------------%
% End of the chain's rule %
%-------------------------%

[dxg]= chain_rule(fs1.x,fs1.G,Dcenter_faceXg,DnormalsXg);


end
