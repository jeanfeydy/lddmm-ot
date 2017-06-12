function options = setdefaultshanso(pars, options)
% set HANSO defaults that are not set by setdefaults
% called only by hanso
%
%   Send comments/bug reports to Michael Overton, overton@cs.nyu.edu,
%   with a subject header containing the string "hanso".
%   Version 2.2, 2016, see GPL license info below.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  HANSO 2.2 Copyright (C) 2010  Michael Overton
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

% this is also in setdefaults, but setdefaults_hanso is called first
% so we need it here too, as pars.nvar is referenced below
if ~isfield(pars, 'nvar')
   error('hanso: input "pars" must have a field "nvar" (number of variables)')
elseif ~isposint(pars.nvar)
   error('hanso: input "pars.nvar" (number of variables) must be a positive integer')
end
nvar = pars.nvar;
% the following isn't needed since it's in setdefaults, but may as well
% include it so error message says "hanso"
if ~isfield(pars, 'fgname')
   error('hanso: input "pars" must have a field "fgname" (name of m-file computing function and gradient)')
end
if ~isfield(options, 'normtol')
    options.normtol = 1e-4;
elseif ~isposreal(options.normtol)
    error('hanso: input "options.normtol" must be a positive scalar')
end
if ~isfield(options, 'evaldist')
    options.evaldist = 1e-4;
elseif ~isposreal(options.evaldist)
    error('hanso: input "options.evaldist" must be a positive scalar')
end
if ~isfield(options, 'samprad')
    % default is NO gradient sampling as of HANSO 2.2: too expensive
    options.samprad = []; % no gradient sampling
else
    % check sampling radii are positive and decreasing
    if any(sort(options.samprad,'descend') ~= options.samprad) || any(options.samprad <= 0)
        error('options.samprad must have decreasing positive entries')
    end
end
% the following are from HANSO 1.0, 1.01
% for backwards compatibility:
if isfield(options, 'phasemaxit')
    if ~isfield(options, 'prtlevel') | options.prtlevel > 0
         fprintf('hanso: "options.phasemaxit" is no longer used in HANSO 2.0\n')
         fprintf(' setting "options.maxit" to "options.phasemaxit(1)"\n\n')
    end
    options.maxit = options.phasemaxit(1); % for BFGS (ignore the others)
end
if isfield(options, 'phasenum')
    if ~isfield(options, 'prtlevel') | options.prtlevel > 0 
         fprintf('hanso: "options.phasenum" is no longer used in HANSO 2.0\n')
         fprintf(' number of BFGS starting points is either 10 or number of columns in "options.x0"\n')
         fprintf(' number of gradient sampling phases is 3, or 0 if pars.nvar > 100, or number of entries in options.gradsamp\n\n')
    end
    % ignore the number of phases specified and use defaults
end