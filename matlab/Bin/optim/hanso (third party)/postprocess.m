function [loc, X, G, w] = postprocess(x, g, dnorm, X, G, w)
% postprocessing of set of sampled or bundled gradients
% if x is not one of the columns of X, prepend it to X and
% g to G and recompute w and dnorm: this can only reduce dnorm
% also set loc.dnorm to dnorm and loc.evaldist to the
% max distance from x to columns of X
% note: w is needed as input argument for the usual case that 
% w is not recomputed but is just passed back to output
%
%   Send comments/bug reports to Michael Overton, overton@cs.nyu.edu,
%   with a subject header containing the string "hanso".
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

for j = 1:size(X,2)
    dist(j) = norm(x - X(:,j));
end
evaldist = max(dist); % for returning
[mindist, indx] = min(dist); % for checking if x is a column of X
if mindist == 0 & indx == 1
    % nothing to do
elseif mindist == 0 & indx > 1
    % this should not happen in HANSO 2.0
    % swap x and g into first positions of X and G
    % might be necessary after local bundle, which is not used in HANSO 2.0
    X(:,[1 indx]) = X(:,[indx 1]);
    G(:,[1 indx]) = G(:,[indx 1]);
    w([1 indx]) = w([indx 1]);
else
    % this cannot happen after BFGS, but it may happen after gradient
    % sampling, for example if max iterations exceeded: line search found a
    % lower point but quit before solving new QP
    % prepend x to X and g to G and recompute w
    X = [x X];
    G = [g G];
    [w,d] = qpspecial(G); % Anders Skajaa's QP code
    dnorm = norm(d);
end
loc.dnorm = dnorm;
loc.evaldist = evaldist;