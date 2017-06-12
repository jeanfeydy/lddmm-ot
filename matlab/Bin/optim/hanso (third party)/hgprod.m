function r = hgprod(H0, g, S, Y)
%  compute the product required by the LM-BFGS method
%  see Nocedal and Wright
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

N = size(S,2);  % number of saved vector pairs (s,y) 
q = g;
for i = N:-1:1
   s = S(:,i);
   y = Y(:,i);
   rho(i) = 1/(s'*y);
   alpha(i) = rho(i)*(s'*q);
   q = q - alpha(i)*y;
end
r = H0*q;
for i=1:N
   s = S(:,i);
   y = Y(:,i);
   beta = rho(i)*(y'*r);
   r = r + (alpha(i)-beta)*s;
end
