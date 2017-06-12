function  res = dcost_varifold(x,y,weight)
% This function compute the derivative wrt X of the cost function used in the OT for discrete measure
% composed with varifold Dirac mass.
%
% Input :
%   X is a discrete varifold measure : a d x 2n matrix where d is the dimension of
%   the ambient space and n is the number of points in the cloud. The first
%   n rows encode the position and en last n rows encode the orientation
%   Y : idem as X.
%
% Output :
%   res : a (d x size(x,2) x size(y,2)) matrix);
% Author : B. Charlier (2017)


[d,nx] = size(x);%nx = nx/2;
[~,ny] = size(y);%ny = ny/2;

d = d/2; %dimension ambient space

%------------------%
%-Cost on position-%
%------------------%

% gradient with respect to x of C(x,y)=1/2*|x-y|^2

nablaC1 = @(x,y)repmat(x,[1 1 size(y,2)]) - ...
    repmat(reshape(y,[size(y,1) 1 size(y,2)]),[1 size(x,2)]);


%---------------------%
%-Cost on orientation-%
%---------------------%

normalsX = x(d+1:2*d,:);
normalsY = y(d+1:2*d,:);

% Compute unit normals
norm_normalsX = sqrt(sum(normalsX .^2,1));
norm_normalsY = sqrt(sum(normalsY .^2,1));

unit_normalsX = normalsX ./  repmat(norm_normalsX,d,1);
unit_normalsY = normalsY ./  repmat(norm_normalsY,d,1);

prs_unit_norm = zeros(nx,ny);
for l=1:d
    prs_unit_norm = prs_unit_norm + (repmat(unit_normalsY(l,:),nx,1).*repmat(unit_normalsX(l,:)',1,ny));
end

dprs_unit_norm =   ( -repmat(unit_normalsX,[1 1 ny]) .* repmat(reshape(prs_unit_norm,[1,nx,ny]),[d,1])...
    + repmat(reshape(unit_normalsY,[d 1 ny]),[1 nx])) ./ reshape(repmat(norm_normalsX,d,ny),[d,nx,ny]);

%unoriented : gradient with respect to x of C(x,y)=2 * (1 - <x/|x| , y/|y|>^2) 
%nablaC2=  -8  .* dprs_unit_norm .* repmat(reshape(prs_unit_norm,[1,nx,ny]),[d,1]);

       
%oriented :  gradient with respect to  (1 - <x/|x| , y/|y|>) 
nablaC2 = - dprs_unit_norm;    

%--------%
%-Result-%
%--------%

% canonical metric (c1 + c2)
res = [weight(1) * nablaC1(x(1:d,:),y(1:d,:));weight(2) * nablaC2];


% Other pseudo metric tricks... (c1 + c1 * c2)
%C1 = @(x,y)squeeze( sum(nablaC1(x,y).^2)/2 );
%C2 =  (2 - 2 * prs_unit_norm .^2);
%res = [ nablaC1(x(1:d,:),y(1:d,:)) .* repmat(reshape( (1 + weight(2) * C2) , [1,nx,ny]),[d,1]) ;...
%	 .5 *weight(2) *  repmat(reshape( C1(x(1:d,:) ,y(1:d,:) ) , [1,nx,ny]),[d,1]).* nablaC2];



end
