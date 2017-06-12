function options = setdefaultsbfgs(pars, options)
%  set defaults for BFGS (in addition to those already set by setdefaults)
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

% line search options
if isfield(options, 'strongwolfe')
    if options.strongwolfe ~= 0 & options.strongwolfe ~= 1
        error('setdefaultsbfgs: input "options.strongwolfe" must be 0 or 1')
    end
else
    % strong Wolfe is very complicated and is bad for nonsmooth functions
    options.strongwolfe = 0;  
end
if isfield(options, 'wolfe1') % conventionally anything in (0,1), but 0 is OK while close to 1 is not
    if ~isreal(options.wolfe1) | options.wolfe1 < 0 | options.wolfe1 > 0.5
        error('setdefaultsbfgs: input "options.wolfe1" must be between 0 and 0.5')
    end
else
    options.wolfe1 = 1e-4; % conventionally this should be positive, and although
                           % zero is usually fine in practice, there are exceptions
end
if isfield(options, 'wolfe2') % conventionally should be > wolfe1, but both 0 OK for e.g. Shor
    if ~isreal(options.wolfe2) | options.wolfe2 < options.wolfe1  | options.wolfe2 >= 1
        error('setdefaultsbfgs: input "options.wolfe2" must be between max(0,options.wolfe1) and 1')
    end
else
    options.wolfe2 = 0.5;  % 0 and 1 are both bad choices
end
if options.strongwolfe
    if options.prtlevel > 0
        if options.wolfe2 > 0
            fprintf('setdefaultsbfgs: strong Wolfe line search selected, but weak Wolfe is usually preferable\n')
            fprintf('(especially if f is nonsmooth)\n')
        else
            fprintf('setdefaultsbfgs: simulating exact line search\n')
        end
    end
    if ~exist('linesch_sw')
        error('"linesch_sw" is not in path: it can be obtained from the NLCG distribution')
    end
else
    if ~exist('linesch_ww')
        error('"linesch_ww" is not in path: it is required for weak Wolfe line search')
    end
end
if isfield(options, 'quitLSfail')
    if options.quitLSfail ~= 0 & options.quitLSfail ~= 1
        error('setdefaultsbfgs: input "options.quitLSfail" must be 0 or 1')
    end
else
    if options.strongwolfe == 1 & options.wolfe2 == 0
        % simulated exact line search, so don't quit if it fails
        options.quitLSfail = 0;
    else  % quit if line search fails
        options.quitLSfail = 1;
    end
end
% other default options
n = pars.nvar;
if isfield(options, 'nvec')
    if ~isnonnegint(options.nvec)
        error('setdefaultsbfgs: input "options.nvec" must be a nonnegative integer')
    end
%%%% elseif n <= 100 THIS WAS IN VERSION 2.02 AND EARLIER BUT IT CAN BE
%%%% VERY CONFUSING FOR A USER: FULL BFGS IS MUCH BETTER WHEN IT IS
%%%% FEASIBLE, ESPECIALLY FOR NONSMOOTH PROBLEMS
else
    options.nvec = 0;  % full BFGS
%%%% else
%%%% options.nvec = 10; % limited memory BFGS
end
if options.nvec == 0 & n > 500 & options.prtlevel > 0
    fprintf('bfgs: when the number of variables is large, consider using the limited memory variant\n');
    fprintf('bfgs: although limited memory may not work as well as full bfgs, especially on nonsmooth problems\n')
    fprintf('bfgs: for limited memory, set options.nvec > 0, for example 10\n');
end 
if isfield(options,'H0')
    % H0 should be positive definite but too expensive to check
    if any(size(options.H0) ~= [n n])
        error('bfgs: input options.H0 must be matrix with order pars.nvar')
    end
    if options.nvec > 0 & ~issparse(options.H0)
        error('bfgs: input "options.H0" must be a sparse matrix when "options.nvec" is positive')
    end
else
    if options.nvec == 0
        options.H0 = eye(n); % identity for full BFGS
    else
        options.H0 = speye(n); % sparse identity for limited memory BFGS
    end
end
if isfield(options, 'scale')
    if options.scale ~= 0 & options.scale ~= 1
        error('setdefaultsbfgs: input "options.scale" must be 0 or 1')
    end
else
    options.scale = 1;
end
    
% note: if f is smooth, superlinear convergence will ensure that termination
% takes place before too many gradients are used in the QP optimality check
% so the optimality check will not be expensive in the smooth case
if isfield(options,'ngrad')
    if ~isnonnegint(options.ngrad)
        error('setdefaultsbfgs: input "options.ngrad" must be a nonnegative integer')
    end
else % note this could be more than options.nvec
     % rationale: it is only towards the end that we start accumulating
     % many gradients, and then they may be needed to veryify optimality
    options.ngrad = min([100, 2*pars.nvar, pars.nvar + 10]);
end
if isfield(options,'evaldist') 
    if ~isposreal(options.evaldist) % corrected Dec 2016
        error('setdefaultsbfgs: input "options.evaldist" must be a positive real scalar')
    end
else
    options.evaldist = 1e-4; 
end