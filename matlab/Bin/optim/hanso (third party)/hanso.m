function [x, f, loc, X, G, w, H] = hanso(pars, options)
%  HANSO: Hybrid Algorithm for Nonsmooth Optimization, Version 2.2
%  Minimization algorithm intended for nonsmooth, nonconvex functions,
%   but also applicable to functions that are smooth, convex or both.
%   Based on BFGS and Gradient Sampling: see below for more details.
%  [X,F] = HANSO(PARS) returns approximate minimizer and function value 
%   of function with PARS.NVAR variables coded in mfile PARS.FGNAME.
%  [X,F,LOC] = HANSO(PARS,OPTIONS) allows specification of options and
%   also returns an approximate Local Optimality Certificate.
%  [X,F,LOC,X,G,W,H] = HANSO(PARS,OPTIONS) returns additional data 
%   supporting the approximate Local Optimality Certificate.
% 
%   Input parameters
%    pars is a required struct, with two required fields
%      pars.nvar: the number of variables
%      pars.fgname: string giving the name of m-file (in single quotes) 
%         that returns the function and its gradient at a given input x, 
%         with call   [f,g] = fgtest(x,pars)  if pars.fgname is 'fgtest'.
%         Any data required to compute the function and gradient may be
%         encoded in other fields of pars. The user does not have to worry
%         about the nondifferentiable case or identify subgradients. 
%         The basic assumption is that the nondifferentiable case arises
%         with probability zero, and in the event that it does occur, it is
%         fine to return the gradient of the function at a nearby point.
%         For example, the user does not have to worry about ties in "max".
%    options is an optional struct, with no required fields
%      options.x0: columns are one or more starting vector of variables, 
%          used to intialize the BFGS phase of the algorithm
%          (default: 10 starting points are generated randomly)
%      options.normtol: termination tolerance for smallest vector in 
%          convex hull of saved gradients
%          (default: 1e-4)
%      options.evaldist: evaluation distance used to determine which
%          gradients to use in this computation 
%          (default: 1e-4)
%      options.maxit: max number of iterations for each BFGS starting point
%          (default: 1000)
%      options.nvec: 0 for full BFGS matrix update in BFGS phase, otherwise 
%           number of vectors to save and use in the limited memory updates
%          (default: 0)
%      options.samprad: sampling radii for gradient sampling
%          (default: []: no gradient sampling) 
%      options.fvalquit: quit if f drops below this value 
%          (default: -inf)
%      options.cpumax: quit if cpu time in seconds exceeds this
%          (default: inf)
%      options.prtlevel: print level, 0 (no printing), 1, or 2 (verbose)
%          (default: 1)
%
%   Output parameters 
%    x: column vector, the best point found
%    f: the value of the function at x
%    loc: local optimality certificate, structure with 2 fields:
%      loc.dnorm: norm of a vector in the convex hull of gradients of the 
%          function evaluated at and near x 
%      loc.evaldist: specifies max distance from x at which these gradients 
%       were evaluated. The smaller loc.dnorm and loc.evaldist are, the
%       more likely it is that x is an approximate local minimizer.
%    X: columns are points where these gradients were evaluated, including x
%    G: the gradients of the function at the columns of X
%    w: vector of positive weights summing to one such that loc.dnorm = ||G*w||
%    H: the final BFGS inverse Hessian approximation, typically 
%        very ill-conditioned if the function is nonsmooth
%
%   Method: 
%      BFGS phase: BFGS is run from multiple starting points, taken from
%       the columns of options.x0, if provided, and otherwise 10 points
%       generated randomly. If the termination test was satisfied at the
%       best point found by BFGS, or if options.samprad == [] (the default)
%       HANSO terminates; otherwise, it continues to:
%      Gradient sampling phases: 1 or more gradient sampling phases are run from 
%       lowest point found, using sampling radii provided in options.samprad.
%       A reasonable choice is [10 1 0.1]*options.evaldist.
%      Termination takes place immediately during any phase if
%       options.cpumax CPU time is exceeded.
%   References: 
%    A.S. Lewis and M.L. Overton, Nonsmooth Optimization via Quasi-Newton
%     Methods, Math Programming, 2012
%    J.V. Burke, A.S. Lewis and M.L. Overton, A Robust Gradient Sampling 
%     Algorithm for Nonsmooth, Nonconvex Optimization
%      SIAM J. Optimization 15 (2005), pp. 751-779
%   Send comments/bug reports to Michael Overton, overton@cs.nyu.edu,
%   with a subject header containing the string "hanso".
%   Version 2.2, 2016, see GPL license info below.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  HANSO 2.2 Copyright (C) 2016  Michael Overton
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
% 
% Similarly, the main change from version 2.1 to 2.2 is that in version 2.1,
% the options.samprad default was [] (no gradient sampling) ONLY if
% pars.nvar <= 100, otherwise it was [10 1 0.1]*options.evaldist, but again
% this can be very confusing for a user.

% parameter defaults
if nargin == 0
   error('hanso: "pars" is a required input parameter')
end
if nargin == 1
   options = [];
end
% call setdefaultshanso first so its defaults take precedence
options = setdefaultshanso(pars, options); % special fields for HANSO
options = setdefaults(pars, options); % this routine is called by other codes too
% the BFGS and line search defaults are set by setdefaultsbfgs, called by bfgs later
cpufinish = cputime + options.cpumax;
normtol = options.normtol;
evaldist = options.evaldist;
fvalquit = options.fvalquit;
prtlevel = options.prtlevel;
if prtlevel > 0  
    fprintf('HANSO Version 2.2, optimizing objective %s over %d variables with options\n', ...
        pars.fgname, pars.nvar')
    disp(options)
end
% BFGS phase with possibly multiple starting points (10 if not provided by user)
[x, f, d, H, iter, info, X, G, w] = bfgs(pars, options); 
if length(f) > 1 % more than one starting point
    [f,indx] = min(f); % throw away all but the best result
    x = x(:,indx);
    d = d(:,indx);
    H = H{indx}; % bug if do this when only one start point: H already matrix
    X = X{indx};
    G = G{indx};
    w = w{indx};
end
dnorm = norm(d);
% the 2nd argument will not be used sinc x == X(:,1) after bfgs
[loc, X, G, w] = postprocess(x, nan, dnorm, X, G, w);
if isnaninf(f)
    if prtlevel > 0
        fprintf('hanso: f is infinite or nan at all starting points\n')
    end
    return
end 
if cputime > cpufinish
    if prtlevel > 0
        fprintf('hanso: cpu time limit exceeded\n')
    end
    if options.prtlevel > 0
        fprintf('hanso: best point found has f = %g with local optimality measure: dnorm = %5.1e, evaldist = %5.1e\n',...
          f, loc.dnorm, loc.evaldist)
    end
    return
end
if f < fvalquit
    if prtlevel > 0
        fprintf('hanso: reached target objective\n')
    end
    if options.prtlevel > 0
        fprintf('hanso: best point found has f = %g with local optimality measure: dnorm = %5.1e, evaldist = %5.1e\n',...
          f, loc.dnorm, loc.evaldist)
    end
    return
end
if dnorm < normtol 
    if prtlevel > 0
        fprintf('hanso: verified optimality within tolerance in bfgs phase\n')
    end
    if options.prtlevel > 0
        fprintf('hanso: best point found has f = %g with local optimality measure: dnorm = %5.1e, evaldist = %5.1e\n',...
          f, loc.dnorm, loc.evaldist)
    end
    return
end
% options.samprad was set in setdefaultshanso (the default is [])
if ~isempty(options.samprad) 
% launch gradient sampling
    f_BFGS = f;
    % save optimality certificate info in case gradient sampling cannot
    % improve the one provided by BFGS
    dnorm_BFGS = dnorm;
    loc_BFGS = loc;
    d_BFGS = d;
    X_BFGS = X;
    G_BFGS = G;
    w_BFGS = w;
    options.x0 = x;
    if options.maxit > 100
        options.maxit = 100; % otherwise gradient sampling is too expensive
    end
    options.nstart = 1; % otherwise gradient sampling will augment with random starts
    options.cpumax = cpufinish - cputime;  % time left
    [x, f, g, dnorm, X, G, w] = gradsamp(pars, options);
    if f == f_BFGS % gradient sampling did not reduce f
        if prtlevel > 0
            fprintf('hanso: gradient sampling did not reduce f below best point found by BFGS\n')
        end
        % use the better optimality certificate
        if dnorm > dnorm_BFGS
            loc = loc_BFGS;
            d = d_BFGS;
            X = X_BFGS;
            G = G_BFGS;
            w = w_BFGS;
        end
    elseif f < f_BFGS
        [loc, X, G, w] = postprocess(x, g, dnorm, X, G, w);
        if prtlevel > 0
            fprintf('hanso: gradient sampling reduced f below best point found by BFGS\n')
        end
    else
        error('hanso: f > f_BFGS: this should never happen') % this should never happen
    end
end
if options.prtlevel > 0
    fprintf('hanso: best point found has f = %g with local optimality measure: dnorm = %5.1e, evaldist = %5.1e\n',...
      f, loc.dnorm, loc.evaldist)
end