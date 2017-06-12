function [x, f, g, dnorm, X, G, w] = gradsamp(pars, options)
%GRADSAMP Gradient sampling algorithm for nonsmooth, nonconvex
%minimization.
%   Intended for nonconvex functions that are continuous everywhere and for 
%   which the gradient can be computed at most points, but which are known 
%   to be nondifferentiable at some points, typically including minimizers.
%
%   Calls:  [x, f, g, dnorm, X, G, w] = gradsamp(pars) 
%    and:   [x, f, g, dnorm, X, G, w] = gradsamp(pars, options)
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
%    options is an optional struct, with no required fields
%       options.x0: each column is a starting vector of variables
%          (default: empty)
%       options.nstart: number of starting vectors, generated randomly
%          if needed to augment those specified in options.x0
%          (default: 10 if options.x0 is not specified)
%       options.samprad: vector of sampling radii (positive and decreasing)
%          (default: [1e-4 1e-5 1e-6]) 
%       options.maxit: max number of iterations per sampling radius 
%          (default: 100) (applies to each starting vector)
%       options.normtol: stopping tolerance on norm(d), where d is vector with
%          smallest norm in the convex hull of the sampled gradients:
%          a scalar or a positive decreasing vector like options.samprad
%          (default: 1e-6) (applies to each starting vector)
%       options.ngrad: number of sampled gradients per iterate
%          (default: min(100, 2*pars.nvar, pars.nvar + 10) 
%       options.fvalquit: quit minimizing if f reaches this target value 
%          (default: -inf) (applies to each starting vector)
%       options.cpumax: quit if cpu time in seconds exceeds this
%          (default: inf) (applies to total running time)
%       options.prtlevel: one of 0 (no printing), 1 (minimal), 2 (verbose)
%          (default: 1)
%
%   Output parameters 
%    x: column k is the best point found by the run from starting point k
%    f: f(k) is the value of the function at x(:,k)
%    g: column k is the gradient at x(:,k)
%    dnorm: dnorm(k) is the norm of a vector in the convex hull of 
%       gradients of the function evaluated at points near x(:,k)
%       (approximately within the final sampling radius).  
%        The smaller dnorm(k) and the closer the points are to x(:,k), 
%        the more confidence one may have that x(:,k) is approximately
%        a local minimizer
%    X: X{k}(:,j), j=1,2,... are points where the gradients were
%        evaluated, usually but not always including x(:,k) = X{k}(:,1) 
%        (X is a cell array, except if there was only one starting point, 
%         X is returned as a matrix)
%    G: G{k}(:,j) is the gradient at X{k}(:,j) (G is a cell array, except
%        if there was only one starting point, G is returned as a matrix)
%    w: w(:,k) is the vector of positive weights summing to one such that
%       d = G{k}*w(:,k), and dnorm(k) is the norm of d
%
%   Reference: J.V. Burke, A.S. Lewis and M.L. Overton,
%    A Robust Gradient Sampling Algorithm for Nonsmooth, Nonconvex Optimization
%    SIAM J. Optimization, 2005
%   This is an updated version of the code.  For the original code used
%   in the experiments reported in the SIOPT paper, see
%   www.cs.nyu.edu/overton/papers/gradsamp/alg/
%
%   Send comments/bug reports to Michael Overton, overton@cs.nyu.edu,
%   with a subject header containing the string "hanso" or "gradsamp".
%   Version 2.0, 2010, see GPL license info below.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  HANSO 2.0 Copyright (C) 2010  Michael Overton
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

if nargin == 0
    error('gradsamp: "pars" is a required input parameter, with fields "nvar" and "fgname"')
end;
if nargin == 1
    options = [];
end
options = setdefaults(pars,options); % set default options
options = setx0(pars,options); % augment options.x0
x0 = options.x0;
nstart = size(x0,2);
nvar = pars.nvar;
maxit = options.maxit;
cpufinish = cputime + options.cpumax;
prtlevel = options.prtlevel;
if ~isfield(options, 'samprad')
    options.samprad = [1e-4 1e-5 1e-6];
else
    osr = options.samprad;
    if  ~isreal(osr) | min(size(osr)) ~= 1 | min(osr) <= 0 | ...
     min(sort(osr) == fliplr(osr)) == 0
        error('gradsamp: input "options.samprad" must be positive and in decreasing order')
    end
end
if ~isfield(options, 'ngrad')% 150 is the max for the free version of MOSEK
    options.ngrad = min([100, 2*nvar, nvar + 10]);  
elseif ~isposint(options.ngrad)
    error('hanso: input "options.ngrad" must be a positive integer')
elseif options.ngrad == 1 & prtlevel > 0
    fprintf('gradsamp: since number of sampled gradients is 1, method reduces to steepest descent\n')
elseif options.ngrad <= pars.nvar & prtlevel > 1
    fprintf('gradsamp: number of sampled gradients <= number of variables\n')
end
for run = 1: nstart
    if prtlevel > 0 & nstart > 1
        fprintf('gradsamp: starting point %d \n', run);
    end
    [f0,g0] = feval(pars.fgname, x0(:,run), pars);
    if isnan(f0) | f0 == inf | maxit == 0
        if isnan(f0) & prtlevel > 0 
            fprintf('gradsamp: function is NaN at initial point\n')
        elseif f0 == inf & prtlevel > 0 
            fprintf('gradsamp: function is infinite at initial point\n')
        elseif maxit == 0 & prtlevel > 0 % useful if just want to evaluate f
            fprintf('gradsamp: max iteration limit is 0, returning initial point\n')
        end
        f(run) = f0;
        x(:,run) = x0(:,run);
        g(:,run) = g0;
        dnorm(:,run) = norm(g0);
        X{run} = x(:,run);
        G{run} = g0;
        w(:,run) = 1;
    else       
        options.cpumax = cpufinish - cputime; % time left
        [x(:,run), f(run), g(:,run), dnorm(run), X{run}, G{run}, w(:,run)] = ...
            gradsamp1run(x0(:,run), f0, g0, pars, options);
    end
    if cputime > cpufinish
        break
    end
end
if nstart == 1
    X = X{1};
    G = G{1};
end