function [dx,dp] = denr_tan_free(templatex,momentum)
%[dx,df,dp,dfunres] = DENR_TAN_FREE(templatex,momentums) computes the gradient
% of the energy functional coded in enr_tan_free.
%
% Input :
%   templatex : cell with the points position
%   momentums : cell with the momentums attached to each point
%
% Output :
%    dx : gradient wrt x
%    dp : gradient wrt momentums
% Author : B. Charlier (2017)

global data objfunc defoc deflag templateG

[nb_obs,nb_match] = size(data);
nlist = diff(deflag); % number of point in each shape

dx = cell(1,nb_match);
dp = cell(nb_obs,1);

templatextotal = cell2mat(templatex'); [n,d] = size(templatextotal);

%---------------------------%
%  gradient of penalization %
%---------------------------%

for l= 1:nb_match
	dx{l} = zeros(size(templatex{1,l}));
end


dX = cell(nb_obs,1);

for sh_ind=1:nb_obs %parallel computations
    
    %sliced variable in parfor
    datac = data(sh_ind,:);
    momentumc = momentum{sh_ind};
    
    %---------------------%
    % gradient dmatchterm %
    %---------------------%

    [shootingx,shootingmom]=forward_tan(templatextotal,momentum{sh_ind},defoc);
    
    dxfinalg= cell(1,nb_match);
    dfg= cell(1,nb_match);
    for l = 1:nb_match
        %load the shooted fshape (== final position)
        templatefinal= struct('x',shootingx{end}(deflag(l)+1:deflag(l+1),:),'f',zeros(size(shootingx{end}(deflag(l)+1:deflag(l+1),:),1),1),'G',templateG{l});
        %load current target
        targetc = struct('x',datac{l}.x,'G',datac{l}.G);
        % gradient of the data attachment term wrt the final position
        [dxfinalg{l}]=dmatchterm(templatefinal,targetc,objfunc{l});
	dxfinalg{l} = dxfinalg{l} .*objfunc{l}.weight_coef_dist ;
	
	%save template final
	export_fshape_vtk(templatefinal,['./',num2str(l),'-results_iter_c.vtk']); 
    end
    
    %at the final position the derivative wrt the moment is 0
    dpfinalg=zeros(n,d);
    % integrate backward the adjoint system to get the gradient at init
    [dxg,dpg]=backward_tan(cell2mat(dxfinalg'),dpfinalg,struct('x',{shootingx},'mom',{shootingmom}),defoc);
    
    %----------------------------%
    % gradient of the prior part %
    %----------------------------%
    
    [dxu,dpu] =  dnormRkhsV(templatextotal,momentumc,defoc);
    
    dXt = cell(1,nb_match);
        
    for l = 1:nb_match
        
        % Normalization in order to get a gradient scale invariant
        dxg(deflag(l)+1:deflag(l+1),:)=dxg(deflag(l)+1:deflag(l+1),:).*objfunc{l}.dgxC;
        dpg(deflag(l)+1:deflag(l+1),:)=dpg(deflag(l)+1:deflag(l+1),:).*objfunc{l}.dgxC;
        dfg{l}=dfg{l}.*objfunc{l}.dgfC;
        
        %-----------------------------------%
        % gradient of penalization (funres) %
        %-----------------------------------%
        dXt{l} = objfunc{1}.weight_coef_pen_p * dxu(deflag(l)+1:deflag(l+1),:) + dxg(deflag(l)+1:deflag(l+1),:);
         
    end
    
    dp{sh_ind} = objfunc{1}.weight_coef_pen_p *dpu +dpg;
    dX{sh_ind} = dXt;
     
end

for i= 1:nb_obs
    
    dx = cellfun(@(y,z) y+z,dx,dX{i},'uniformoutput',0);
    
end



for l=1:nb_match
    
    % Normalization of the gradient
    dx{l}=dx{l}*nlist(l);
    
    
end

dp= cellfun(@(y) y/n,dp,'UniformOutput',0);

end
