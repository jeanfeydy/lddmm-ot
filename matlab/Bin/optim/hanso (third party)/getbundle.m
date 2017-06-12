function [xbundle, gbundle] = getbundle(x, g, samprad, N, pars);
%  get bundle of N-1 gradients at points near x, in addition to g,
%  which is gradient at x and goes in first column
%  intended to be called by gradsampfixed
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

m = length(x);
xbundle(:,1) = x;
gbundle(:,1) = g;
for k = 2:N  % note the 2
   xpert = x + samprad*(rand(m,1) - 0.5); % uniform distribution
   [f,grad] = feval(pars.fgname, xpert, pars);
   count = 0;
   while isnaninf(f) | isnaninf(grad)  % in particular, disallow infinite function values
       xpert = (x + xpert)/2;     % contract back until feasible
       [f,grad] = feval(pars.fgname, xpert, pars);
       count = count + 1;
       if count > 100 % should never happen, but just in case
           error('too many contractions needed to find finite f and grad values')
       end
   end; % discard function values
   xbundle(:,k) = xpert;
   gbundle(:,k) = grad;   
end
