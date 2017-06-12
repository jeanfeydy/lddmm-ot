function [dxinit,dpinit]=backward_tan(dxfinal,dpfinal,shooting,defo)
% [dxinit,dpinit]=BACKWARD(dxfinal,dpfinal,shooting,defo) performs the
% backward integration for the tangential Hamiltonian flow.
%
% Input :
%   dxfinal : gradient in x (point) wrt the final position
%   dpfinal : gradient in p (momenta) wrt the final momenta
%   shooting : a  structure containing evolution path of the points (shooting.x{i} is a cell where i ranges from 1 to defo.nb_euler_steps)
%       and momenta (shooting.mom{i} is a cell where i ranges from 1 to defo.nb_euler_steps) attached to the template during the shooting
%   defo : structure containing the parameters of deformations (kernel size, nstep... and so on)
%
% Ouput
%   dxinit : gradient vector wrt the inital position and momenta (n x d) column vector.
%   dpinit : gradient vector  wrt the inital momenta (nd x 1) column vector.
%
% See also : ddHrtP_tan, forward_tan, dHr_tan
% Author : B. Charlier (2017)

% Step size
h=1/defo.nb_euler_steps;

% Initiatialze dxinit (output) to dxfinal before backward integration
dxinit=dxfinal;
dpinit = dpfinal;

for i=defo.nb_euler_steps+1:-1:2
   % backward midpoint method :    
    [dx2,dp2]=fddh2(shooting.x{i},shooting.mom{i},dxinit,dpinit,defo,-h/2);
    [dx3,dp3]=fddh2((shooting.x{i-1}+shooting.x{i})/2,(shooting.mom{i-1}+shooting.mom{i})/2,dx2,dp2,defo,-h);

    dxinit=(dx3-dx2)+dxinit;
    dpinit=(dp3-dp2)+dpinit;
end

end

function [nPx,nPp]=fddh2(x,p,Px,Pp,defo,h)
% Basic Euler step 

    [dPx,dPp] = ddHrtP_tan(x,p,Px,Pp,defo);
  
    nPx=Px+h*dPx;
    nPp=Pp+h*dPp;

end
