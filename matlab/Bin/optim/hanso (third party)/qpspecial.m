function [x,d,q,info] = qpspecial(G,varargin)
% Call:
% [x,d,q,info] = qpspecial(G,varargin)
%
% Solves the QP
%
%    min     q(x)  = || G*x ||_2^2 = x'*(G'*G)*x
%    s.t.  sum(x)  = 1
%              x  >= 0
%
% The problem corresponds to finding the smallest vector
% (2-norm) in the convex hull of the columns of G
%
% Inputs:
%     G            -- (M x n double) matrix G, see problem above
%     varargin{1}  -- (int) maximum number of iterates allowed
%                     If not present, maxit = 100 is used
%     varargin{2}  -- (n x 1 double) vector x0 with initial (FEASIBLE) iterate.
%                     If not present, (or requirements on x0 not met) a
%                     useable default x0 will be used
%
% Outputs:
%     x       -- Optimal point attaining optimal value
%     d = G*x -- Smallest vector in the convex hull
%     q       -- Optimal value found = d'*d
%     info    -- Run data:
%                info(1) =
%                   0 = everything went well, q is optimal
%                   1 = maxit reached and final x is feasible. so q
%                       might not be optimal, but it is better than q(x0)
%                   2 = something went wrong
%                info(2) = #iterations used
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  HANSO 2.0 Copyright (C) 2010  Anders Skajaa
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

echo  = 0;                    % set echo = 1 for printing 
                              % (for debugging). Otherwise 0. 
                              %
[m,n] = size(G);              % size of problem
if ~(m*n>0)                   % in this case
    fprintf(['qpspecial:',... % G is empty, so nothing we can do
        ' G is empty.\n']);   % exit with warning 
    info = [2,0];             % info(1) = 2;
    x = []; d = []; q = inf;  % and empty structs
    return;                   % and optimal value is inf
end                           %
                              %
e     = ones(n,1);            % vector of ones
                              %
if nargin > 1                 % set defauls
    maxit = varargin{1};      % maximal # iterations
    maxit = ceil(maxit);      % in case of a non-integer input
    maxit = max(maxit,5);     % always allow at least 5 iterations
else                          %
    maxit = 100;              % default is 100
end                           % which is always plenty
if nargin > 2                 % 
    x  = varargin{2};         % if x0 is specified
    x  = x(:);                % if given as row instead of col
    nx = size(x,1);           % check that size is right
    if any(x<0) || nx~=n      % use it, unless it is 
        x = e;                % infeasible in the ineq 
    end                       % constraints. 
else                          % use it, otherwise
    x = e;                    % use (1,1,...,1)
end                           % which is an interior point
                              %
idx   = (1:(n+1):(n^2))';     % needed many times
Q     = G'*G;                 % Hessian in QP
z     = x;                    % intialize z
y     = 0;                    % intialize y
eta   = 0.9995;               % step size dampening
delta = 3;                    % for the sigma heuristic
mu0   = (x'*z) / n;           % first my
tolmu = 1e-5;                 % relative stopping tolerance, mu
tolrs = 1e-5;                 % and residual norm
kmu   = tolmu*mu0;            % constant for stopping, mu
nQ    = norm(Q,inf)+2;        % norm of [Q,I,e]
krs   = tolrs*nQ;             % constant for stopping, residuals
ap    = 0; ad = 0;            % init steps just for printing
if echo > 0                   % print first line
    fprintf(['k    mu   ',... %
   '     stpsz     res\n',... %
   '-----------------',...    %
   '-----------------\n']);   % 
end                           %
                              %
for k = 1:maxit               %
                              %
    r1 = -Q*x + e*y + z;      % residual
    r2 = -1 + sum(x);         % residual
    r3 = -x.*z;               % slacks
    rs = norm([r1;r2],inf);   % residual norm
    mu = -sum(r3)/n;          % current mu
                              %
                              % printing (for debugging)
    if echo > 0
        fprintf('%-3.1i %9.2e %9.2e %9.2e \n',...
            k,mu/mu0,max(ap,ad),rs/nQ);
    end
                              % stopping
    if mu < kmu               % mu must be small
        if rs < krs           % primal feas res must be small
            info = [0,k-1];   % in this case, all went well
            break;            % so exit with info = 0
        end                   % so exit loop
    end                       %
                              %
    zdx     = z./x;           % factorization
    QD      = Q;              %  
    QD(idx) = QD(idx) + zdx;  % 
    [C,ef]  = chol(QD);       % C'*C = QD
    if ef > 0                 % safety to catch possible
        info = [2,k];         % problems in the choleschy  
        break;                % in this case, 
    end                       % break with info = 2
    KT      = C'\e;           % K'   = (C')^(-1) * e
    M       = KT'*KT;         % M    = K*K'
                              %
    r4  = r1+r3./x;           % compute approx 
    r5  = KT'*(C'\r4);        % tangent direction
    r6  = r2+r5;              % using factorization 
    dy  = -r6/M;              % from above
    r7  = r4 + e*dy;          %
    dx  = C\(C'\r7);          %
    dz  = (r3 - z.*dx)./x;    %
                              %
    p   = -x ./ dx;           % Determine maximal step 
    ap  = min(min(p(p>0)),1); % possible in the 
    if isempty(ap)            % approx tangent direction
        ap = 1;               % here primal step size
    end                       % 
    p   = -z ./ dz;           % here dual step size
    ad  = min(min(p(p>0)),1); % 
    if isempty(ad)            % using different step sizes
        ad = 1;               % in primal and dual improves
    end                       % performance a bit
                              %  
    muaff = ((x + ap*dx)'*... % heuristic for 
             (z + ad*dz))/n;  % the centering parameter
    sig   = (muaff/mu)^delta; %
                              %
    r3  = r3 + sig*mu;        % compute the new corrected
    r3  = r3 - dx.*dz;        % search direction that now 
    r4  = r1+r3./x;           % includes the appropriate
    r5  = KT'*(C'\r4);        % amount of centering and 
    r6  = r2+r5;              % mehrotras second order 
    dy  = -r6/M;              % correction term (see r3).
    r7  = r4 + e*dy;          % we of course reuse the 
    dx  = C\(C'\r7);          % factorization from above
    dz  = (r3 - z.*dx)./x;    %
                              %
    p   = -x ./ dx;           % Determine maximal step 
    ap  = min(min(p(p>0)),1); % possible in the 
    if isempty(ap)            % new direction
        ap = 1;               % here primal step size
    end                       % 
    p   = -z ./ dz;           % here dual step size
    ad  = min(min(p(p>0)),1); % 
    if isempty(ad)            % 
        ad = 1;               % 
    end                       % 
                              % update variables
    x   = x + eta * ap * dx;  % primal
    y   = y + eta * ad * dy;  % dual multipliers
    z   = z + eta * ad * dz;  % dual slacks 
                              %
end                           % end main loop
                              %
if k == maxit                 % if reached maxit
    info = [1,k];             % set info(1) = 1
end                           %
x = max(x,0);                 % project x onto R+
x = x/sum(x);                 % so that sum(x) = 1 exactly                        
d = G*x;                      % and set other output 
q = d'*d;                     % variables using best 
                              % found x
                              %
if echo > 0                   % printing
    str = 'optimal';          % status string to print
    if info(1)==1             % in the last line
        str = 'maxit reached';%
    elseif info(1)==2         %
        str = 'failed';       %
    end                       %
    fprintf(['---------',...  % print last line
    '-------------------',... %
    '------\n',...            %     
    ' result: %s \n',...      %
    '-------------------',... %
    '---------------\n'],str);%
end                           %
    



