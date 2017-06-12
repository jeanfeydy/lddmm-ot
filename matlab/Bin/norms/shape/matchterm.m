function g= matchterm(fshape1,fshape2,objfun)
% g= MATCHTERM(fshape1,fshape2,objfun) computes the distance between
% fshape1 and fshape2 wrt various norms. Options 
% are given by the structure objfun.
%
%Input
%   fshape1 : fshape structure
%   fshape2 : fshape structure
%   objfun : structure containing the parameters : 
%Output
% g = real number
% Author : B. Charlier (2017)


% Some checks
if (size(fshape1.x,2) ~= size(fshape2.x,2)) || (size(fshape1.G,2) ~= size(fshape2.G,2))
    error('fshapes should be in the same space')
end

switch objfun.distance

    case 'wasserstein'
        G = @fshape_wasserstein_distance;
        
end


g = G(fshape1,fshape2,objfun);

end

