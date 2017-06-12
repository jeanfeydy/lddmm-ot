function [ENR,dist,penp] = enr_tan_free(templatex,momentums)
% enr_tan_free(templatex,momentum) computes the energy functional
% in the tangential and free framework.
%
% Input :
%   templatex : cell with the points position
%   momentums : cell with the momentums attached to each point
%
%Output :
% ENR: energy (a number)
% dist,penp : terms composing the energy
% Author : B. Charlier (2017)


global data objfunc defoc deflag templateG

tstart =tic;

%---------------
%  indiv. terms
%---------------

[nb_obs,nb_match] = size(data);

enrg = zeros(nb_obs,1);
enru = zeros(nb_obs,1);

templatextotal = cell2mat(templatex');

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
        templatefinal= struct('x',shootingx{end}(deflag(l)+1:deflag(l+1),:),'G',templateG{l});
        
        %load current target
        targetc = struct('x',datac{l}.x,'G',datac{l}.G);

        enrg(sh_ind) = enrg(sh_ind) + objfunc{l}.weight_coef_dist * objfunc{l}.gC * matchterm(templatefinal,targetc,objfunc{l});
        
    end
end


%---------------
% Energy term
%---------------

dist=sum(enrg);
penp=sum(enru);
ENR = penp + dist ;

if nargout ==1
    fprintf('enr %4.2e: dist %4.2e, pen_p %4.2e, time %f\n', ENR, dist,penp,toc(tstart))
end

end
