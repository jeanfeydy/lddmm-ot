function [noptim,mesg] = set_optim_option(optim,objfun,list_of_variables,enr)
% This function check the optim structure and set default values if needed.
% Author : B. Charlier (2017)


noptim = optim;


noptim = setoptions(noptim,'method','gradDesc',{'gradDesc','bfgs'});
noptim = setoptions(noptim,noptim.method,[]);


switch lower(noptim.method)
    case 'graddesc'
        noptim.gradDesc = set_gradDesc_defaults(noptim.gradDesc,objfun,list_of_variables,enr);
        
    case 'bfgs'
        noptim.bfgs = set_bfgs_defaults(noptim.bfgs,list_of_variables,enr);
        
end


% save message and display
mesg = [sprintf('\n----------- Optimization parameters -----------\n' ),...
    dispstructure(noptim)];

fprintf('%s',mesg);




end

function opt = set_gradDesc_defaults(opt,objfun,list_of_variables,enr_name)


opt = setoptions(opt,'step_increase',1.2);
opt = setoptions(opt,'step_decrease',.5);

opt = setoptions(opt,'kernel_size_signal_reg',0);
opt = setoptions(opt,'kernel_size_geom_reg',0);

switch objfun{1}.distance
    case 'kernel'
        opt = setoptions(opt,'max_nb_iter',50 * ones(1,size(objfun{1}.kernel_distance.kernel_size_signal,2)));
        
        nb_run = length(opt.max_nb_iter);
        nb_match = length(objfun);
        
        if ~isequal(cell2mat(cellfun(@(x) length(x.kernel_distance.kernel_size_signal),objfun,'UniformOutput',0)),nb_run*ones(1,nb_match)) ...
                || ~isequal(cell2mat(cellfun(@(x) length(x.kernel_distance.kernel_size_geom),objfun,'UniformOutput',0)),nb_run*ones(1,nb_match))...
                ||  ( (strcmp(objfun{1}.kernel_distance.kernel_grass,'gaussian_oriented')|| strcmp(objfun{1}.kernel_distance.kernel_grass,'gaussian_unoriented') ) && ~isequal(cell2mat(cellfun(@(x) length(x.kernel_distance.kernel_size_grass),objfun,'UniformOutput',0)),nb_run*ones(1,nb_match)))
            error('All kernel sizes objfun.sigmaXX and optim.max_nb_iter must have the same length');
        end
    
    case 'wasserstein'
        opt = setoptions(opt,'max_nb_iter',50 * ones(1,size(objfun{1}.wasserstein_distance.epsilon,2)));

end

opt = setoptions(opt,'save_template_evolution',0);
opt = setoptions(opt,'min_step_size',1e-10);
opt = setoptions(opt,'min_fun_decrease',1e-4);

for i =1:length(list_of_variables)
    opt = setoptions(opt,['step_size_',list_of_variables{i}],'auto');
end

opt = setoptions(opt,'list_of_variables',list_of_variables);
opt = setoptions(opt,'enr_name',enr_name);
end

function opt = set_bfgs_defaults(opt,list_of_variables,enr_name)



opt = setoptions(opt, 'nvec', 20); % BFGS memory
opt = setoptions(opt, 'maxit', 50);
opt = setoptions(opt, 'prtlevel',2);
opt = setoptions(opt, 'normtol', eps);
opt = setoptions(opt, 'tol',eps);
opt = setoptions(opt,'list_of_variables',list_of_variables);
opt = setoptions(opt,'enr_name',enr_name);


end

