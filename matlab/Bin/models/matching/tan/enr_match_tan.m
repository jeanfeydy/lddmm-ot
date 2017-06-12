function [ENR,dENR] = enr_match_tan(variables_vec)
% Simple wrapper function that can be called by Hanso bfgs.
% Author : B. Charlier (2017)

global deflag templatexc templatefc

n = deflag(end) ;
d = size(templatexc{1},2);


mom_cell = {reshape(variables_vec(1:d*n),n,d)};

ENR = enr_tan_free(templatexc,mom_cell);


[~,dnom] = denr_tan_free(templatexc,mom_cell) ;


dENR = [cell2mat(dnom )];
dENR = dENR(:);

end


