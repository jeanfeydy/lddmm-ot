function [dxg]= dmatchterm(fshape1,fshape2,objfun)
% g= DMATCHTERM(fshape1,fshape2,objfun) computes the distance between
% fshape1 and fshape2 wrt various norms. Options 
% are given by the structure objfun.
%
%Input
%   fshape1 : a fshape structure
%   fshape2 : a fshape structure
%   objfun : structure containing the parameters
%Output
% dxg : derivative wrt x (matrix n x d) 
% dfg : derivative wrt f (vector n x 1)
% Author : B. Charlier (2017)

% Some checks
if (size(fshape1.x,2) ~= size(fshape2.x,2)) || (size(fshape1.G,2) ~= size(fshape2.G,2))
    error('fshapes should be in the same space')
end


switch objfun.distance
    case 'wasserstein'
        DG = @dfshape_wasserstein_distance;
end



[dxg] = DG(fshape1,fshape2,objfun);

end
