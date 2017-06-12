function [res] = area(pts,tri)
% compute the areas of a set of triangles
%
% Input:
%   pts : list of vertices (n x d matrix)
%   tri : list of edges (T x M matrix where M==2 for curve and M==3 for surface)
%
% Output
%   res : area of each cell (T x 1 column vector);
% Author : B. Charlier (2017)


m = 1 ./ factorial(size(tri,2)-1);
res = sqrt(sum(pVectors(pts,tri) .^2,2)) .* m;

end

