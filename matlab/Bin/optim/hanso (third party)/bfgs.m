function [x, f, d, H, iter, info, X, G, w, fevalrec, xrec, Hrec] = bfgs(pars, options)
%BFGS The BFGS quasi-Newton minimization algorithm, Version 2.2, 2016
%   Basic call:[x, f, d] = bfgs(pars) 
%   Full call: [x, f, d, H, iter, info, X, G, w, fevalrec, xrec, Hrec] = bfgs(pars,options)
%   Input parameters
%    pars is a required struct, with two required fields
%      pars.nvar: the number of variables
%      pars.fgname: string giving the name of function (in single quotes) 
%         that returns the function and its gradient at a given input x, 
%         with call   [f,g] = fgtest(x,pars)  if pars.fgname is 'fgtest'.
%         Any data required to compute the function and gradient may be
%         encoded in other fields of pars.
%    options is an optional struct, with no required fields
%      options.x0: each column is a starting vector of variables
%          (default: empty)
%      options.nstart: number of starting vectors, generated randomly
%          if needed to augment those specified in options.x0
%          (default: 10 if options.x0 is not specified)
%      options.maxit: max number of iterations
%          (default 1000) (applies to each starting vector)
%      options.nvec: 0 for full BFGS matrix update, otherwise specifies 
%           number of vectors to save and use in the limited memory updates
%          (default: 0)
%      options.H0: 
%          for full BFGS: initial inverse Hessian approximation
%           (must be positive definite, but this is not checked)
%          for limited memory BFGS: same, but applied every iteration
%           (must be sparse in this case)
%          (default: identity matrix, sparse in limited memory case)
%      options.scale: 
%          for full BFGS: 1 to scale H0 at first iteration, 0 otherwise
%          for limited memory BFGS: 1 to scale H0 every time, 0 otherwise
%          (default: 1)
%      options.ngrad: number of gradients willing to save and use in
%           solving QP to check optimality tolerance on smallest vector in
%           their convex hull; see also next two options 
%          (default: min(100, 2*pars.nvar, pars.nvar + 10)
%          (1 is recommended if and only if f is known to be smooth)
%      options.normtol: termination tolerance on d: smallest vector in
%           convex hull of up to options.ngrad gradients
%          (default: 1e-6) 
%      options.evaldist: the gradients used in the termination test
%           qualify only if they are evaluated at points  approximately 
%           within distance options.evaldist of x
%          (default: 1e-4) 
%      options.fvalquit: quit if f drops below this value 
%          (default: -inf) 
%      options.xnormquit: quit if norm(x) exceeds this value
%          (default: inf)
%      options.cpumax: quit if cpu time in secs exceeds this 
%          (default: inf) (applies to total running time)
%      options.strongwolfe: 0 for weak Wolfe line search (default)
%                           1 for strong Wolfe line search
%          (strong Wolfe line search is not recommended for use with
%           BFGS; it is very complicated and bad if f is nonsmooth;
%           however, it can be useful to simulate an exact line search)
%      options.wolfe1: first Wolfe line search parameter 
%          (ensuring sufficient decrease in function value, default: 1e-4)
%          (should be > 0 in theory; usually but not always 0 is ok in practice)
%      options.wolfe2: second Wolfe line search parameter 
%          (ensuring algebraic increase (weak) or absolute decrease (strong)
%           in projected gradient, default: 0.5)
%          (important in theory and practice that this is not 0 or 1, 
%           except that it can be set to 0 if an exact line search is to be
%           simulated, using options.strongwolfe = 1)
%      options.quitLSfail: 1 if quit when line search fails, 0 otherwise
%          (default: 1, except if options.strongwolfe = 1 and
%           options.wolfe2 = 0, simulating exact line search)
%          (0 is potentially useful if f is not numerically continuous)
%      options.prtlevel: one of 0 (no printing), 1 (minimal), 2 (verbose)
%          (default: 1)
%
%   Output parameters: 
%    all return information on the runs for each starting vector
%    x: the final iterates
%    f: the final function values 
%    d: the final smallest vectors in the convex hull of the saved gradients 
%     at termination (the final gradient if options.ngrad == 1)
%    H: final BFGS inverse Hessian approximation matrices
%     (useful for full BFGS update only, guaranteed to be symmetric)
%     (multiple of identity matrix if limited memory updates were used)
%    iter: number of iterations
%    info: reason for termination:
%     0: tolerance on smallest vector in convex hull of saved gradients met
%     1: max number of iterations reached
%     2: f reached target value
%     3: norm(x) exceeded limit
%     4: cpu time exceeded limit
%     5: f is inf or nan at initial point
%     6: direction not a descent direction due to rounding error
%     7: line search bracketed minimizer but Wolfe conditions not satisfied
%     8: line search did not bracket minimizer: f may be unbounded below
%    X: iterates where saved gradients were evaluated (see below) 
%    G: saved gradients used for computation of smallest vector in convex hull 
%      of gradients at points near final x
%    w: weights giving the smallest vector in the convex hull of the saved
%      gradients
%    fevalrec: record of all function values evaluated in all line searches, 
%      including the final accepted values (nans if options.strongwolfe = 1)
%    xrec: record of all x iterates 
%    Hrec: record of all H iterates
%     (not symmetrized, may not be symmetric because of rounding error)
%    Note: if there is more than one starting vector, then:
%      f, iter, info are vectors of length options.nstart
%      x, d are matrices of size pars.nvar by options.nstart
%      H, X, G, w, xrec, Hrec are cell arrays of length options.nstart, and 
%      fevalrec is a cell array of cell arrays
%    Thus, for example, d(:,i) = G{i}*w{i}, for i = 1,...,options.nstart
%
%   BFGS is normally used for optimizing smooth, not necessarily convex, 
%   functions, for which the convergence rate is generically superlinear.
%   But it also works very well for functions that are nonsmooth at their  
%   minimizers, typically with a linear convergence rate and a final 
%   inverse Hessian approximation that is very ill conditioned, as long 
%   as a weak Wolfe line search is used. This version of BFGS will work
%   well both for smooth and nonsmooth functions and has a stopping 
%   criterion that applies for both cases, described above.
%   Reference:  A.S. Lewis and M.L. Overton, Nonsmooth Optimization via 
%     Quasi-Newton Methods, Math Programming, 2012
%   Send comments/bug reports to Michael Overton, overton@cs.nyu.edu,
%   with a subject header containing the string "hanso" or "bfgs".
%   Version 2.2, 2016, see GPL license info below.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  HANSO 2.2 Copyright (C) 2015  Michael Overton
%%  This program is free software: you can redistribute it and/or modify
%%  it under the terms of the GNU General Public License as published by
%%  the Free Software Foundation, either version 3 of the License, or
%%  (at your option) any later version.
%%
%%  This program is distributed in the hope that it will be useful,
%%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%%  GNU General Public License for more details.
%%
%%  You should have received a copy of the GNU General Public License
%%  along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Version note: the main change from version 2.02 to version 2.1 is that in
% version 2.02, the options.nvec default was 0 ONLY if pars.nvar <= 100, 
% otherwise it was 10. However, this can be very confusing for a user
% because, especially for nonsmooth problems, limited memory BFGS does not
% perform nearly as well as full BFGS. So, in version 2.1, the default is
% always options.nvec = 0, implying full BFGS. Instead, a warning message
% is printed if the number of variables exceeds 500 and options.prtlevel>0.

% No change in bfgs going from version 2.1 to 2.2, except correcting an error
% in the comments about the default value for options.wolfe1 and correcting 
% a typo in setdefaultsbfgs

% parameter defaults
if nargin == 0
   error('bfgs: "pars" is a required input parameter')
end
if nargin == 1
   options = [];
end
options = setdefaults(pars, options);  % set most default options
options = setx0(pars, options); % augment options.x0 randomly
x0 = options.x0;
nstart = size(x0,2);
cpufinish = cputime + options.cpumax;
fvalquit = options.fvalquit;
xnormquit = options.xnormquit;
prtlevel = options.prtlevel;
% set other options
options = setdefaultsbfgs(pars, options); 
for run = 1:nstart
    if prtlevel > 0 & nstart > 1
        fprintf('bfgs: starting point %d\n', run)
    end
    options.cpumax = cpufinish - cputime; % time left
    if nargout > 9
        [x(:,run), f(run), d(:,run), HH, iter(run), info(run), X{run}, G{run}, w{run}, ...
           fevalrec{run}, xrec{run}, Hrec{run}] = bfgs1run(x0(:,run), pars, options);
    elseif nargout > 6 % avoid computing fevalrec, xrec, Hrec which are expensive as they grow inside the main loop
        [x(:,run), f(run), d(:,run), HH, iter(run), info(run), X{run}, G{run}, w{run}] = bfgs1run(x0(:,run), pars, options);
    else % avoid computing unnecessary cell arrays 
        [x(:,run), f(run), d(:,run), HH, iter(run), info(run)] = bfgs1run(x0(:,run), pars, options);
    end
    % HH should already be exactly symmetric as of version 2.02, but does no harm
    H{run} = (HH + HH')/2; 
    if cputime > cpufinish % in an older version we also quite if these held: | f < fvalquit | norm(x) > xnormquit
        break
    end
end
if nstart == 1 % no point returning cell arrays of length 1
    H = H{1};
    if nargout > 9
        fevalrec = fevalrec{1};
        xrec = xrec{1};
        Hrec = Hrec{1};  % don't symmetrize
    end
    if nargout > 6
        X = X{1};
        G = G{1};
        w = w{1};
    end
end