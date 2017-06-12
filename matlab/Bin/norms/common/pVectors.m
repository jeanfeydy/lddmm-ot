function res = pVectors(pts,tri)
% computes (a representation of) the p-vector (ie tangent vector for curve and normals for surfaces). 
%
% Input:
%  pts: list of vertices (n x d matrix)
%  tri: list of edges (T x M matrix of indices)
%
% Output:
%  res: list of p-vectors (d x M matrix)
% Author : B. Charlier (2017)


M = size(tri,2);
d = size(pts,2);

if (M==2) % curves
    
    res=pts(tri(:,2),:)-pts(tri(:,1),:);
    
elseif (M==3) && (d==3) % surfaces

    u=pts(tri(:,2),:)-pts(tri(:,1),:); 
    v=pts(tri(:,3),:)-pts(tri(:,1),:);
    
    res =[u(:,2).*v(:,3)-u(:,3).*v(:,2),...
          u(:,3).*v(:,1)-u(:,1).*v(:,3),...
          u(:,1).*v(:,2)-u(:,2).*v(:,1)];

elseif (M==1) % points

    res = repmat( 1./size(tri,1) ,size(tri,1),1)  ;
  
elseif (M==3) && (d==2)% simplexes

    u=pts(tri(:,2),:)-pts(tri(:,1),:); 
    v=pts(tri(:,3),:)-pts(tri(:,1),:);

    res =  u(:,1).*v(:,2) - u(:,2).*v(:,1);

elseif (M==4) && (d==3)% simplexes

    u=pts(tri(:,2),:)-pts(tri(:,1),:); 
    v=pts(tri(:,3),:)-pts(tri(:,1),:);
    w=pts(tri(:,4),:)-pts(tri(:,1),:);

    res =  u(:,1).*v(:,2).*w(:,3) + v(:,1).*w(:,2).*u(:,3) + w(:,1).*u(:,2).*v(:,3)...
         - u(:,3).*v(:,2).*w(:,1) - v(:,3).*w(:,2).*u(:,1) - w(:,3).*u(:,2).*v(:,1); %det with Sarrus formula

end

end
