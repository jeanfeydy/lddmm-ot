function [x,summary] = perform_bfgs(Grad, x, options)

% perform_bfgs - wrapper to HANSO code
%
%   [f, R, info] = perform_bfgs(Grad, f, options);
%
%   Grad should return (value, gradient)
%   f is an initialization
%   options.niter is the number of iterations.
%   options.bfgs_memory is the memory for the hessian bookeeping.
%   R is filled using options.repport which takes as input (f,val).
% Author : B. Charlier (2017)


n = length(x);
pars.nvar = n;
pars.fgname = @(f,pars) eval([Grad,'(f)']);

options.x0 = x;

tstart = tic;
[x, energy, ~, ~, iter, info, ~, ~, ~, fevalrec, xrec, ~] = bfgs(pars,options);


info_exit =  {'tolerance on smallest vector in convex hull of saved gradients met',...
'max number of iterations reached',...
'f reached target value',...
'norm(x) exceeded limit',...
'cpu time exceeded limit',...
'f is inf or nan at initial point',...
'direction not a descent direction due to rounding error',...
'line search bracketed minimizer but Wolfe conditions not satisfied',...
'line search did not bracket minimizer: f may be unbounded below'};


summary.bfgs.list_of_energy = [fevalrec{1}(1),cellfun(@(x) x(end), fevalrec)]; 
summary.bfgs.exit_flags=info_exit{info+1};
summary.bfgs.nb_of_iterations=iter;
summary.bfgs.computation_time = toc(tstart);
summary.bfgs.xrec = xrec;


end
