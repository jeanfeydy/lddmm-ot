function options = setx0(pars,options)
% set columns of options.x0 randomly if not provided by user
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

nvar = pars.nvar;
if ~isfield(options, 'x0')
    options.x0 = [];
end
if isempty(options.x0)
    if isfield(options, 'nstart')
        if ~isposint(options.nstart)
            error('setx0: input "options.nstart" must be a positive integer when "options.x0" is not provided')
        else
            options.x0 = randn(nvar, options.nstart);
        end
    else
        options.x0 = randn(nvar, 10);
    end
else
    if size(options.x0,1) ~= nvar
        error('setx0: input "options.x0" must have "pars.nvar" rows')
    end
    if isfield(options, 'nstart')
        if ~isnonnegint(options.nstart)
            error('setx0: input "options.nstart" must be a nonnegative integer')
        elseif options.nstart < size(options.x0,2)
            error('setx0: "options.nstart" is less than number of columns of "options.x0"')
        else % augment vectors in options.x0 with randomly generated ones
            nrand = options.nstart - size(options.x0,2);
            options.x0 = [options.x0  randn(nvar, nrand)];
        end
    end % no else part, options.x0 is as provided
end