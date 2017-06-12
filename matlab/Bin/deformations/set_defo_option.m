function [ndefo,mesg] = set_defo_option(defo,list_of_variables)
% This function check the defo structure and set default values if needed.
% Author : B. Charlier (2017)

is_algo_met =(sum(strcmp(list_of_variables(:),'pf')) >0);

ndefo = defo;

ndefo = setoptions(ndefo,'method','matlab',{'cuda','matlab','grid'});
    
if (strcmp(defo.method,'grid')) && ( ~isfield(defo,'gridratio') || (defo.gridratio < 0) || (defo.gridratio >1) )
	ndefo.gridratio = .2;
	%fprintf('deformation : gridratio set to %f\n',ndefo.gridratio)
end

ndefo = setoptions(ndefo,'nb_euler_steps',10);
ndefo = setoptions(ndefo,'kernel_size_mom');

if is_algo_met
    ndefo = setoptions(ndefo,'weight_coef_pen_p',1);
    ndefo = setoptions(ndefo,'weight_coef_pen_pf',1);
end

% save message and display
mesg = [sprintf('\n----------- Deformation parameters -----------\n' ),...
        evalc('disp(orderfields(ndefo))')];
fprintf('%s',mesg);

end
