function [dxg2]=chain_rule(x,tri,DXg,DXig)
%
% Computation of the gradient with respect to x by distributing the previous gradients
% on points of the initial shape. (Chain's rule). There is two versions of the same
% code somewhat equivalent : one using matlab for loop (that is not so bad
% due to JIT precompilation process) and one using built in matlab function
% accumarray.
%
% Input:
%  x: list of vertices (n x d matrix)
%  G: list of edges (T x M matrix)
%  DXg: derivative wrt X (center of the cells: T x d matrix)
%  DXig: derivative wrt Xi (p-vectors: T x d matrix)
%  Dtfg: derivative wrt tf (signal at center of the cells: T x1 colunm vector)
%
% Output:
%  dxg: derivative wrt x (n x d matrix)
%  dfg: derivative wrt f (n x 1 colunm vector)
% Author : B. Charlier (2017)



[nx,d]=size(x);
[~,M] = size(tri);

if M==1 % point cloud case  chain's rule stops here

    dxg2 = DXg;
    return
end



if (d==2)
    U2 =  [accumarray(tri(:),repmat(DXg(:,1),M,1),[nx,1],[],0),...
           accumarray(tri(:),repmat(DXg(:,2),M,1),[nx,1],[],0)] / M;
elseif (d==3)
    U2 =  [accumarray(tri(:),repmat(DXg(:,1),M,1),[nx,1],[],0),...
           accumarray(tri(:),repmat(DXg(:,2),M,1),[nx,1],[],0),...
           accumarray(tri(:),repmat(DXg(:,3),M,1),[nx,1],[],0)] / M;
end


if (M==2) % curve case
    if (d==2)
        dxg2 = U2 + [accumarray(tri(:),[-DXig(:,1);DXig(:,1)] ,[nx,1],[],0),...
                     accumarray(tri(:),[-DXig(:,2);DXig(:,2)] ,[nx,1],[],0) ];
    elseif (d==3)
        dxg2 = U2 + [accumarray(tri(:),[-DXig(:,1);DXig(:,1)] ,[nx,1],[],0),...
                     accumarray(tri(:),[-DXig(:,2);DXig(:,2)] ,[nx,1],[],0),...
                     accumarray(tri(:),[-DXig(:,3);DXig(:,3)] ,[nx,1],[],0) ];
    end
    
elseif (M==3) && (d==3) % surface case
    
    Xa=x(tri(:,1),:);
    Xb=x(tri(:,2),:);
    Xc=x(tri(:,3),:);
    

    [dU1,dU2,dU3,dV1,dV2,dV3] = dcross((Xb-Xa),(Xc-Xa),DXig/2);
    
    dxg2 = U2 + [accumarray(tri(:),[-dU1-dV1;+dU1 ;+dV1 ],[nx,1],[],0),...
                 accumarray(tri(:),[-dU2-dV2;+dU2 ;+dV2 ],[nx,1],[],0),...
                 accumarray(tri(:),[-dU3-dV3;+dU3 ;+dV3 ],[nx,1],[],0)];

elseif (M==4) && (d==3) % simplexes case

    Xa=x(tri(:,1),:);
    Xb=x(tri(:,2),:);
    Xc=x(tri(:,3),:);
    Xd=x(tri(:,4),:);
    
    u=(Xb-Xa);v=(Xc-Xa);w=(Xd-Xa);

    %[dU1,dU2,dU3,dV1,dV2,dV3,dW1,dW2,dW3] =  dcrosss(u,v,w,H);
    H = DXig/6;
    
    dU1 = ( v(:,2) .* w(:,3) - v(:,3) .* w(:,2) ).* H;
    dU2 = ( v(:,3) .* w(:,1) - v(:,1) .* w(:,3) ).* H;
    dU3 = ( v(:,1) .* w(:,2) - v(:,2) .* w(:,1) ).* H;
   
    dV1 = ( w(:,2) .* u(:,3) - w(:,3) .* u(:,2) ).* H;
    dV2 = ( w(:,3) .* u(:,1) - w(:,1) .* u(:,3) ).* H;
    dV3 = ( w(:,1) .* u(:,2) - w(:,2) .* u(:,1) ).* H;
    
    dW1 = ( u(:,2) .* v(:,3) - u(:,3) .* v(:,2) ).* H;
    dW2 = ( u(:,3) .* v(:,1) - u(:,1) .* v(:,3) ).* H;
    dW3 = ( u(:,1) .* v(:,2) - u(:,2) .* v(:,1) ).* H;
    

    
    dxg2 = U2 + [accumarray(tri(:),[-dU1-dV1-dW1 ;+dU1 ;+dV1 ; dW1],[nx,1],[],0),...
                  accumarray(tri(:),[-dU2-dV2-dW2 ;+dU2 ;+dV2 ; dW2],[nx,1],[],0),...
                  accumarray(tri(:),[-dU3-dV3-dW3 ;+dU3 ;+dV3 ; dW3],[nx,1],[],0)];
    

    
elseif (M==3) && (d==2) % simplexes case

    Xa=x(tri(:,1),:);
    Xb=x(tri(:,2),:);
    Xc=x(tri(:,3),:);
    
    U=(Xb-Xa);V=(Xc-Xa);
    
    H = DXig/2;
    
    dU1 =   V(:,2) .* H;
    dU2 =  -V(:,1) .* H;
    
    dV1 =  -U(:,2) .* H ;
    dV2 =   U(:,1) .* H;
    
    dxg2 = U2 + [accumarray(tri(:),[-dU1-dV1;+dU1 ;+dV1 ],[nx,1],[],0)...
                 accumarray(tri(:),[-dU2-dV2;+dU2 ;+dV2 ],[nx,1],[],0)];
    
end

end
