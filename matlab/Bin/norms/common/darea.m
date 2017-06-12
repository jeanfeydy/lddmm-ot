function [dArea] = darea(pts,tri)
% compute the derivative of the areas of a set of triangles
%
% Input:
%   pts : nb_points x d matrix
%   tri : nb_tri x M matrix (M==2 for curve and M==3 for surface)
%
% Output
%   dArea : (M* nb_tri) x d matrix;
% Author : B. Charlier (2017)


[~,M] = size(tri);
[~,d] = size(pts);

N = pVectors(pts,tri);
N = bsxfun(@rdivide,N,sqrt(sum(N.^2,2)));

if (M==2) && (d<=3)

    dArea = [-N;N];
 elseif (M==3) && (d==3)
     
     E21 =  (pts(tri(:,2),:) - pts(tri(:,1),:));
     E31 =  (pts(tri(:,3),:) - pts(tri(:,1),:));
     
     [dG1,dG2,dG3,dD1,dD2,dD3] = dcross(E21,E31,N);

     dArea =   .5*  [[-dG1-dD1;+dG1 ;+dD1 ],...
                     [-dG2-dD2;+dG2 ;+dD2 ],...
                     [-dG3-dD3;+dG3 ;+dD3 ]];
end

end
