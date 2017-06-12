function [X,Xi] = fcatoms(pts,tri)
% [X,tf,Xi] = FCATOMS(pts,f,tri,signal_type) compute the dirac representation of meshes (1D or 2D)
%
% Input :
%  pts :  matrix with vertices (one column per dimension, matrix nxd)
%  tri : connectivity matrix. if tri is of size (M x 2) this is a 1d current 
%  	(so each line of tri contains the two indexes of the points defining 
%  	an oriented segment), if tri is (M x 3) this to be considered 
%  	as a 2d current (so each line of tri contains the three indexes of 
%  	the points defining an oriented triangle).
%  f : column vector of functional values (n x1)
%  signal_type : optional string. If  type_signal=='face', f is assumed to be
%  the values of the signal at the center of the faces else f is assumed to
%  be the values of the signal at the vertices (ie pts)
%
% Output
% X : the matrix of the centers of the faces   (M x d)
% Xi: is the matric of p-vectors (tangent (1d current) or normal 2d current) (M x 2 or 3)
% Author : B. Charlier (2017)


[T,M]=size(tri);
d = size(pts,2);

%normals 
m = factorial(M-1);
Xi = pVectors(pts,tri) / m; % normals or tangent vectors

%centers
if M==1
    X = pts;
else
    X=sum(reshape(pts(tri(:),:)',d,T,M ),3)'/M; % center of the faces
end

end

