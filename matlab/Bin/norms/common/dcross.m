function [dG1,dG2,dG3,dD1,dD2,dD3] = dcross(G,D,H)
% Differential of (. \wedge .) at points (G,D) and applied to H. In short : d_(G,D) (G\wedge D) (H)
% Author : B. Charlier (2017)

    dG1 =  ( -D(:,3) .* H(:,2) +  D(:,2) .* H(:,3));
    dG2 =  ( +D(:,3) .* H(:,1) -  D(:,1) .* H(:,3));
    dG3 =  ( -D(:,2) .* H(:,1) +  D(:,1) .* H(:,2));
    
    dD1 =  ( +G(:,3) .* H(:,2) -  G(:,2) .* H(:,3));
    dD2 =  ( -G(:,3) .* H(:,1) +  G(:,1) .* H(:,3));
    dD3 =  ( +G(:,2) .* H(:,1) -  G(:,1) .* H(:,2));
    
end
