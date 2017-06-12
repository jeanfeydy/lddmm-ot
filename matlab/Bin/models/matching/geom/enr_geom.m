function [ENR,dENR] = enr_geom(momentums_vec)
% Author : B. Charlier (2017)

global deflag 

mom_cell = {reshape(momentums_vec,deflag(end),[])};

ENR = enr_match_geom(mom_cell);

dENR = cell2mat( denr_match_geom(mom_cell) );
dENR = dENR(:);

end

function [ENR] = enr_match_geom(momentums)
% ENR_MATCH_GEOM(templatex,templatef,momentum) computes the energy functional
% for a pure geometric matching.
%
% Input :
%   momentums : cell with the momentums attached to each point
%
% Input as global vars :
%   data objfunc defoc deflag templateG templatexc templatefc
%
%Output :
% ENR: energy (a number)
% Author : B. Charlier (2017)


global data objfunc defoc deflag templateG templatexc templatefc
tstart = tic;
%----------------%
%  indiv. terms  %
%----------------%

[nb_obs,nb_match] = size(data);

enrg = zeros(nb_obs,1);
enru = zeros(nb_obs,1);

templatextotal = cell2mat(templatexc');

for sh_ind=1:nb_obs %parallel computations

    %sliced variable in parfor
    datac = data(sh_ind,:);
    momentumc = momentums{sh_ind};

    % compute the energy of the deformation
    enru(sh_ind) = objfunc{1}.weight_coef_pen_p *objfunc{1}.mC*scalarProductRkhsV(momentumc,templatextotal,defoc);   %  objfunc{1}.mC est commun!

    % shoot the template
    [shootingx,~]=forward_tan(templatextotal,momentumc,defoc);

    for l = 1:nb_match
        
        %load the deformed fshape number l (== final position)
        templatefinal= struct('x',shootingx{end}(deflag(l)+1:deflag(l+1),:),'f',templatefc{l},'G',templateG{l});
        
        %load current target
        targetc = struct('x',datac{l}.x,'f',datac{l}.f,'G',datac{l}.G);

        enrg(sh_ind) = enrg(sh_ind) + objfunc{l}.weight_coef_dist * objfunc{l}.gC * matchterm(templatefinal,targetc,objfunc{l});

    end
end


%-------------%
% Energy term %
%-------------%

dist=sum(enrg);
penp=sum(enru);
ENR = penp + dist;

%if nargout ==1
    fprintf('enr %4.2e: dist %4.2e, pen_p %4.2e, time %f\n', ENR, dist,penp,toc(tstart))
%end

end



function [dp] = denr_match_geom(momentum)
% DENR_MATCH_GEOM(templatex,templatef,momentum) computes the gradient of
% energy functional for a pure geometric matching.
%
% Input :
%   momentums : cell with the momentums attached to each point
%
% Input as global vars :
%   data objfunc defoc deflag templateG templatexc templatefc
%
%Output :
%    dp : gradient wrt momentums
% Author : B. Charlier (2017)


global data objfunc defoc deflag templateG templatexc templatefc

[nb_obs,nb_match] = size(data);

dp = cell(nb_obs,1);

templatextotal = cell2mat(templatexc'); [n,d] = size(templatextotal);


for sh_ind=1:nb_obs %parallel computations
    
    %sliced variable in parfor
    datac = data(sh_ind,:);
    momentumc = momentum{sh_ind};
    
    %---------------------%
    % gradient dmatchterm %
    %---------------------%

    [shootingx,shootingmom]=forward_tan(templatextotal,momentumc,defoc);
    
    dxfinalg= cell(1,nb_match);

    for l = 1:nb_match
        %load the shooted fshape (== final position)
        templatefinal= struct('x',shootingx{end}(deflag(l)+1:deflag(l+1),:),'f',templatefc{l},'G',templateG{l});
        %load current target
        targetc = struct('x',datac{l}.x,'f',datac{l}.f,'G',datac{l}.G);
        % gradient of the data attachment term wrt the final position
        dxfinalg{l}=dmatchterm(templatefinal,targetc,objfunc{l});
	dxfinalg{l} = dxfinalg{l} .*objfunc{l}.weight_coef_dist ;
	
	%save template final
	export_fshape_vtk(templatefinal,['./',num2str(l),'-results_iter_c.vtk']); 
    end
    
    %at the final position the derivative wrt the moment is 0
    dpfinalg=zeros(n,d);
    % integrate backward the adjoint system to get the gradient at init
    [~,dpg]=backward_tan(cell2mat(dxfinalg'),dpfinalg,struct('x',{shootingx},'mom',{shootingmom}),defoc);
    
    %----------------------------%
    % gradient of the prior part %
    %----------------------------%
    
    [~,dpu] =  dnormRkhsV(templatextotal,momentumc,defoc);
    
    
    for l = 1:nb_match
        
        % Normalization in order to get a gradient scale invariant
        dpg(deflag(l)+1:deflag(l+1),:)=dpg(deflag(l)+1:deflag(l+1),:).*objfunc{l}.dgxC;
    end
    
    dp{sh_ind} = objfunc{1}.weight_coef_pen_p *dpu +dpg;
     
end

dp= cellfun(@(y) y/n,dp,'UniformOutput',0);

end
