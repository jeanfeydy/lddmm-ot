function g= fshape_wasserstein_distance(fs1,fs2,objfun)
% FSHAPE_KERNEL_DISTANCE(templatefinal,target,objfun) computes kernel based
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
%   g : a real number.
% Author : B. Charlier (2017)

d=size(fs1.x,2);
m=size(fs1.G,2)-1;

% discretize the fshapes
[center_faceX,normalsX]=fcatoms(fs1.x,fs1.G);
[center_faceY,normalsY]=fcatoms(fs2.x,fs2.G);

options = objfun.wasserstein_distance;

if min(m, d-m) ==0 % for points clouds or simplexes dim or codim == 0 : some simplifications occurs
    
    x=center_faceX';
    y=center_faceY';
    
    mu = fs1.G;
    nu = fs2.G;
    
    %only needed for matlab version. See built-in function for cuda version.    
    nablaC = @(x,y,~)repmat(x,[1 1 size(y,2)]) - ...
        repmat(reshape(y,[size(y,1) 1 size(y,2)]),[1 size(x,2)]);
    C = @(x,y,~)squeeze( sum(nablaC(x,y).^2)/2 );% C(x,y)=1/2*|x-y|^2
 
elseif min(m,d-m) ==1 % for curves or surface dim or codim ==1;
    
    mu = area(fs1.x,fs1.G);
    nu = area(fs2.x,fs2.G);
    
    x=[center_faceX';normalsX'];
    y=[center_faceY';normalsY'];
    
    %only needed for matlab version. See built-in function for cuda version.    
    C = @(X,Y) cost_varifold(X,Y,options.weight_cost_varifold);

end


switch lower(options.method)
    case 'matlab'
        
        [~,~,~,~,g,~] = sinkhorn_log(mu,nu,C(x,y),options.epsilon,options);
        g =  g(end);
    case 'cuda'
        
        [~,~,~,g,~] = sinkhorn_log_cuda(mu,nu,x,y,options.epsilon,options);
        
end

end
