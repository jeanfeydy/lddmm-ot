function [x, f, g, dnorm, X, G, w] = gradsamp1run(x0, f0, g0, pars, options);
% repeatedly run gradient sampling minimization, for various sampling radii
% return info only from final sampling radius
% intended to be called by gradsamp only
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

samprad = options.samprad;
cpufinish = cputime + options.cpumax;
for choice = 1:length(samprad)
    options.cpumax = cpufinish - cputime; % time left
    [x, f, g, dnorm, X, G, w, quitall] = ...
        gradsampfixed(x0, f0, g0, samprad(choice), pars, options);
    % it's not always the case that x = X(:,1), for example when the max
    % number of iterations is exceeded: this is mentioned in the
    % comments for gradsamp
    if quitall % terminate early
        return  
    end
    % get ready for next run, with lower sampling radius
    x0 = x;   % start from where previous one finished,
                           % because this is lowest function value so far
    f0 = f;
    g0 = g;
end
