function [x_evol,p_evol]=forward_tan(x_init,p_init,defo,tf)
% [x_evol,p_evol]=FORWARD(x_init,p_init,defo,final_time) compute 
% Forward integration of the Hamiltonian flow from initial coupled
% configuration of points/momentums.
%
% Input :
%  x_init : initial coordinates of the points (position) in a (n x d) matrix.
%  p_init : initial momentums in a (n x d) matrix.
%  defo : structure containing the parameters of deformations (kernel_size_mom,method,nstep,...)
%  final_time : final time (optional, and fixed by default to 1)
%
% Output
%  x_evol : a cell list containing evolution path of positions ( points_evol{i} is a n x d matrix and i ranges from 1 to defo_options.nstep+1)
%  p_evol : a cell list containing evolution path of momentums ( nomentums_evol{i} is a n x d matrix and i ranges from 1 to defo_options.nstep+1)
%
% See also : backward_tan, dHr_tan, ddHrtP_tan
% Author : B. Charlier (2017)


if nargin == 3
    tf=1;
end

x_evol=cell(1,defo.nb_euler_steps+1);
p_evol=cell(1,defo.nb_euler_steps+1);

dt=tf/defo.nb_euler_steps;

x_evol{1} =deal(x_init);
p_evol{1} = deal(p_init);

for i=1:defo.nb_euler_steps
    
    % Midpoint method 
    [x2,p2]=fdh(x_evol{i},p_evol{i},defo,dt/2);
    [x3,p3]=fdh(x2,p2,defo,dt);
    
    x_evol{i+1} = x3 - x2 + x_evol{i};
    p_evol{i+1}  = p3  - p2  + p_evol{i};
    
end

end

function [nx,np]=fdh(x,p,defo,h)
% This fonction implements an elementary Euler Step
%
% Inputs :
%   x: is a (n x d) matrix containing the points.
%   p: is a (n x d) matrix containing the momentums.
%   defo: is a structure of deformations.
%   h is the time step.

% Outputs
%   nx : (n x d) matrix containing the new points.
%   np :(n x d) matrix containing the new the momentums.

    %here f = dHr
    [dp,dx]= dHr_tan(x,p,defo);

    nx=x+h*dx;
    np=p-h*dp;

end
