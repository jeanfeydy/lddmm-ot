function [x, f, d, H, iter, info, X, G, w, fevalrec, xrec, Hrec] = bfgs1run(x0, pars, options)
% Version 2.02, 2013
% make a single run of BFGS from one starting point
% intended to be called by bfgs.m
% outputs: 
%    x: final iterate
%    f: final function value
%    d: final smallest vector in convex hull of saved gradients
%    H: final inverse Hessian approximation
%    iter: number of iterations
%    info: reason for termination
%     0: tolerance on smallest vector in convex hull of saved gradients met
%     1: max number of iterations reached
%     2: f reached target value
%     3: norm(x) exceeded limit
%     4: cpu time exceeded limit
%     5: f or g is inf or nan at initial point
%     6: direction not a descent direction (because of rounding)
%     7: line search bracketed minimizer but Wolfe conditions not satisfied
%     8: line search did not bracket minimizer: f may be unbounded below 
%    X: iterates where saved gradients were evaluated
%    G: gradients evaluated at these points
%    w: weights defining convex combination d = G*w
%    fevalrec: record of all function evaluations in the line searches
%    xrec: record of x iterates
%    Hrec: record of H iterates
%   Send comments/bug reports to Michael Overton, overton@cs.nyu.edu,
%   with a subject header containing the string "hanso" or "bfgs".
%   Version 2.02, 2013, see GPL license info below.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  HANSO 2.02 Copyright (C) 2013  Michael Overton
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

n = pars.nvar;
fgname = pars.fgname;
normtol = options.normtol;
fvalquit = options.fvalquit;
xnormquit = options.xnormquit;
cpufinish = cputime + options.cpumax;
maxit = options.maxit;
nvec = options.nvec;
prtlevel = options.prtlevel;
strongwolfe = options.strongwolfe;
wolfe1 = options.wolfe1;
wolfe2 = options.wolfe2;
quitLSfail = options.quitLSfail;
ngrad = options.ngrad;
evaldist = options.evaldist;
H0 = options.H0;
H = H0; % sparse for limited memory BFGS 
scale = options.scale;
x = x0;
[f,g] = feval(fgname, x, pars);
d = g;
G = g;
X = x; 
nG = 1;
w = 1;
dnorm = norm(g);
if nvec > 0 % limited memory BFGS
    S = [];
    Y = [];
end
iter = 0;
if nargout > 9
    % so outputs defined if quit immediately
    fevalrec{1} = nan; % cell array
    xrec = nan*ones(n,1); % not cell array
    Hrec{1} = nan; % cell array
end
if isnaninf(f) % better not to generate an error return
    if prtlevel > 0
        fprintf('bfgs: f is infinite or nan at initial iterate\n')
    end
    info = 5;
    return
elseif isnaninf(g)
    if prtlevel > 0
        fprintf('bfgs: gradient is infinite or nan at initial iterate\n')
    end
    info = 5;
    return
elseif dnorm < normtol
    if prtlevel > 0
        fprintf('bfgs: tolerance on gradient satisfied at initial iterate\n')
    end
    info = 0;
    return
elseif f < fvalquit
    if prtlevel > 0
        fprintf('bfgs: below target objective at initial iterate\n')
    end
    info = 2;
    return
elseif norm(x) > xnormquit
    if prtlevel > 0
        fprintf('bfgs: norm(x) exceeds specified limit at initial iterate\n')
    end
    info = 3;
    return
end
for iter = 1:maxit
    if nvec == 0 % full BFGS
        p = -H*g;
    else % limited memory BFGS
        p = -hgprod(H, g, S, Y);  % not H0, as in previous version
    end
    gtp = g'*p;
    if gtp >= 0 | isnan(gtp) % in rare cases, H could contain nans
       if prtlevel > 0
          fprintf('bfgs: not descent direction, quit at iteration %d, f = %g, dnorm = %5.1e\n',...
              iter, f, dnorm)
       end
       info = 6;
       return
    end
    gprev = g;  % for BFGS update
    if strongwolfe 
        % strong Wolfe line search is not recommended
        % except to simulate an exact line search
        % function values are not returned, so set fevalrecline to nan
        fevalrecline = nan;
        [alpha, x, f, g, fail] = ...
            linesch_sw(x, f, g, p, pars, wolfe1, wolfe2, fvalquit, prtlevel);
        if wolfe2 == 0 % exact line search: increase alpha slightly to get 
                       % to other side of any discontinuity in nonsmooth case
            increase = 1e-8*(1 + alpha); % positive if alpha = 0
            x = x + increase*p;
            if prtlevel > 1
                fprintf(' exact line sch simulation: slightly increasing step from %g to %g\n', alpha, alpha + increase)
            end
            [f,g] = feval(pars.fgname, x, pars);
        end
    else % weak Wolfe line search is the default
        [alpha, x, f, g, fail, notused, notused2, fevalrecline] = ...
                linesch_ww(x, f, g, p, pars, wolfe1, wolfe2, fvalquit, prtlevel);
    end
    % for the optimality check:
    % discard the saved gradients iff the new point x is not sufficiently
    % close to the previous point and replace them by new gradient 
    if alpha*norm(p) > evaldist
        nG = 1;
        G = g;
        X = x;
    % otherwise add new gradient to set of saved gradients, 
    % discarding oldest if already have ngrad saved gradients
    elseif nG < ngrad
        nG = nG + 1;
        G =  [g G];
        X = [x X];
    else % nG = ngrad
        G = [g G(:,1:ngrad-1)];
        X = [x X(:,1:ngrad-1)];
    end
    % optimality check: compute smallest vector in convex hull of qualifying 
    % gradients: reduces to norm of latest gradient if ngrad == 1, and the
    % set must always have at least one gradient: could gain efficiency
    % here by updating previous QP solution
    if nG > 1
        [w,d] = qpspecial(G); % Anders Skajaa code for this special QP
    else
        w = 1;
        d = g; 
    end
    dnorm = norm(d);
    if nargout > 9
        xrec(:,iter) = x;
        fevalrec{iter} = fevalrecline; % function vals computed in line search
        Hrec{iter} = H;
    end
    if prtlevel > 1
        nfeval = length(fevalrecline);
        fprintf('bfgs: iter %d: nfevals = %d, step = %5.1e, f = %g, nG = %d, dnorm = %5.1e\n', ...
            iter, nfeval, alpha, f, nG, dnorm)
    end
    if f < fvalquit % this is checked inside the line search
        if prtlevel > 0
            fprintf('bfgs: reached target objective, quit at iteration %d \n', iter)
        end
        info = 2;
        return
    elseif norm(x) > xnormquit % this is not checked inside the line search
        if prtlevel > 0
            fprintf('bfgs: norm(x) exceeds specified limit, quit at iteration %d \n', iter)
        end
        info = 3;
        return
    end
    if fail == 1 % line search failed (Wolfe conditions not both satisfied)
        if ~quitLSfail
            if prtlevel > 1
                fprintf('bfgs: continue although line search failed\n')
            end
        else % quit since line search failed
            if prtlevel > 0
                fprintf('bfgs: quit at iteration %d, f = %g, dnorm = %5.1e\n', iter, f, dnorm)
            end
            info = 7;
            return
        end
    elseif fail == -1 % function apparently unbounded below
        if prtlevel > 0
           fprintf('bfgs: f may be unbounded below, quit at iteration %d, f = %g\n', iter, f)
        end
        info = 8;
        return
    end
    if dnorm <= normtol
        if prtlevel > 0 
            if nG == 1 
                fprintf('bfgs: gradient norm below tolerance, quit at iteration %d, f = %g\n', iter, f')
            else
                fprintf('bfgs: norm of smallest vector in convex hull of gradients below tolerance, quit at iteration %d, f = %g\n', iter, f')
            end
        end
        info = 0;
        return
    end
    if cputime > cpufinish
        if prtlevel > 0
            fprintf('bfgs: cpu time limit exceeded, quit at iteration %d\n', iter)
        end
        info = 4;
        return
    end
    s = alpha*p;
    y = g - gprev;
    sty = s'*y;    % successful line search ensures this is positive
    if nvec == 0   % perform rank two BFGS update to the inverse Hessian H
        if sty > 0 
            if iter == 1 & scale
                % for full BFGS, Nocedal and Wright recommend scaling I before 
                % the first update only
                H = (sty/(y'*y))*H; 
            end
            % for formula, see Nocedal and Wright's book
            %M = I - rho*s*y', H = M*H*M' + rho*s*s', so we have
            %H = H - rho*s*y'*H - rho*H*y*s' + rho^2*s*y'*H*y*s' + rho*s*s'
            % note that the last two terms combine: (rho^2*y'Hy + rho)ss'
            rho = 1/sty;
            Hy = H*y;
            rhoHyst = (rho*Hy)*s';  
            % old version: update may not be symmetric because of rounding 
            % H = H - rhoHyst' - rhoHyst + rho*s*(y'*rhoHyst) + rho*s*s';  
            % new in version 2.02: make H explicitly symmetric
            % also saves one outer product
            % in practice, makes little difference, except H=H' exactly
            ytHy = y'*Hy; % could be < 0 if H not numerically pos def
            sstfactor = max([rho*rho*ytHy + rho,  0]);
            sscaled = sqrt(sstfactor)*s;
            H = H - (rhoHyst' + rhoHyst) + sscaled*sscaled';
            % alternatively add the update terms together first: does
            % not seem to make significant difference
            % update = sscaled*sscaled' - (rhoHyst' + rhoHyst);
            % H = H + update;
        else % should not happen unless line search fails, and in that case should normally have quit
            if prtlevel > 1
                fprintf('bfgs: sty <= 0, skipping BFGS update at iteration %d \n', iter)
            end
        end
    else % save s and y vectors for limited memory update
        s = alpha*p;
        y = g - gprev;
        if iter <= nvec
            S = [S s];
            Y = [Y y];
        else % could be more efficient here by avoiding moving the columns
            S = [S(:,2:nvec) s];
            Y = [Y(:,2:nvec) y];
        end
        if scale 
            H = ((s'*y)/(y'*y))*H0;  % recommended by Nocedal-Wright
        end
    end
end % for loop
if prtlevel > 0
    fprintf('bfgs: %d iterations reached, f = %g, dnorm = %5.1e\n', maxit, f, dnorm)
end
info = 1; % quit since max iterations reached