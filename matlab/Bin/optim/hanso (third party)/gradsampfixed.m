function [x, f, g, dnorm, X, G, w, quitall] = ...
    gradsampfixed(x0, f0, g0, samprad, pars, options)
%  gradient sampling minimization with fixed sampling radius
%  intended to be called by gradsamp1run only
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

fgname = pars.fgname;
prtlevel = options.prtlevel;
if prtlevel > 0
    fprintf('gradsamp: sampling radius = %7.1e,',samprad);
end
if prtlevel > 1
    fprintf('\n')
end
x = x0;
f = f0;
g = g0;
X = x;
G = g;
w = 1;
quitall = 0;
maxit = options.maxit;
normtol = options.normtol;
ngrad = options.ngrad;
fvalquit = options.fvalquit;
cpufinish = cputime + options.cpumax;
dnorm = inf;
for iter = 1:maxit
    % evaluate gradients at randomly generated points near x
    % first column of Xnew and Gnew are respectively x and g
    [Xnew, Gnew] = getbundle(x, g, samprad, ngrad, pars); 
    % solve QP subproblem
    [wnew,dnew] = qpspecial(Gnew); % Anders Skajaa specialized QP solver
    dnew = -dnew; % this is a descent direction
    gtdnew = g'*dnew;  % gradient value at current point
    dnormnew = norm(dnew);
    if dnormnew < dnorm % for returning, may not be the final one
        dnorm = dnormnew;
        X = Xnew;
        G = Gnew;
        w = wnew;
    end
    if dnormnew < normtol    
        % since dnormnew is first to satisfy tolerance, it must equal dnorm
        if prtlevel > 0
            fprintf('  tolerance met at iter %d, f = %g, dnorm = %5.1e\n', ...
               iter, f, dnorm);
        end 
        return 
    elseif gtdnew >= 0 | isnan(gtdnew)
        if prtlevel > 0 % dnorm, not dnormnew, which may be bigger
            fprintf('  not descent direction, quit at iter %d, f = %g, dnorm = %5.1e\n', ...
               iter, f, dnorm);
        end
        return
    end
    % note that dnew is NOT normalized, but we set second Wolfe 
    % parameter to 0 so that sign of derivative must change
    % and this is accomplished by expansion steps when necessary,
    % so it does not seem necessary to normalize d
    wolfe1 = 0;
    wolfe2 = 0;  
    [alpha, x, f, g, fail] = ...
         linesch_ww(x, f, g, dnew, pars, wolfe1, wolfe2, fvalquit, prtlevel); 
    if prtlevel > 1 % since this is printed every iteration we print dnormnew here
        fprintf('  iter %d: step = %5.1e, f = %g, dnorm = %5.1e\n',...
            iter, alpha, f, dnormnew)
    end
    if f < fvalquit
        if prtlevel > 0
            fprintf('  reached target objective, quit at iter %d \n', iter)
        end
        quitall = 1;
        return
    end
    % if fail == 1 % Wolfe conditions not both satisfied, DO NOT quit,
    % because this typically means gradient set not rich enough and we
    % should continue sampling
    if fail == -1 % function apparently unbounded below
        if prtlevel > 0
           fprintf('  f may be unbounded below, quit at iter %d, f = %g\n',...
               iter, f)
        end
        quitall = 1;
        return
    end
    if cputime > cpufinish
        if prtlevel > 0
            fprintf('  cpu time limit exceeded, quit at iter %d\n', iter)
        end
        quitall = 1;
        return
    end
end
if prtlevel > 0
   fprintf('  %d iters reached, f = %g, dnorm = %5.1e\n', maxit, f, dnorm);
end