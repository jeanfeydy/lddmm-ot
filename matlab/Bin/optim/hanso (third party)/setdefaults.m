function options = setdefaults(pars,options)
%  call: options = setdefaults(pars,options)
%  check that fields of pars and options are set correctly and
%  set basic default values for options that are common to various 
%  optimization methods, including bfgs and gradsamp
%   Send comments/bug reports to Michael Overton, overton@cs.nyu.edu,
%   with a subject header containing the string "hanso" or "bfgs".
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
if nargin < 2
    options = [];
end
if ~isfield(pars, 'nvar')
   error('setdefaults: input "pars" must have a field "nvar" (number of variables)')
elseif ~isposint(pars.nvar)
   error('setdefaults: input "pars.nvar" (number of variables) must be a positive integer')
end
if ~isfield(pars, 'fgname')
   error('setdefaults: input "pars" must have a field "fgname" (name of m-file computing function and gradient)')
end
if isfield(options, 'maxit')
    if ~isnonnegint(options.maxit)
        error('setdefaults: input "options.maxit" must be a nonnegative integer')
    end
else
    options.maxit = 1000;
end
if isfield(options, 'normtol')
    if ~isposreal(options.normtol)
        error('setdefaults: input "options.normtol" must be a positive real scalar')
    end
else
    options.normtol = 1.0e-6;
end
if isfield(options, 'fvalquit')
    if ~isreal(options.fvalquit)|~isscalar(options.fvalquit)
        error('setdefaults: input "options.fvalquit" must be a real scalar')
    end
else
    options.fvalquit = -inf;
end
if isfield(options, 'xnormquit')
    if ~isreal(options.xnormquit)|~isscalar(options.xnormquit)
        error('setdefaults: input "options.fvalquit" must be a real scalar')
    end
else
    options.xnormquit = inf;
end
if ~isfield(options, 'cpumax')
    options.cpumax = inf;
end
if ~isfield(options, 'prtlevel')
    options.prtlevel = 1;
end