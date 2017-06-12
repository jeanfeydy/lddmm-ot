function  res = cost_varifold(X,Y,weight)
% This function compute the cost function used in the OT for discrete measure
% composed with varifold Dirac mass.
%
% Input :
%   X is a discrete varifold measure : a d x 2n matrix where d is the dimension of
%   the ambient space and n is the number of points in the cloud. The first
%   n rows encode the position and en last n rows encode the orientation
%   Y : idem as X.
%
% Output :
%   res : a nx x ny matrix of positive real numbers.
% Author : B. Charlier (2017)


[d,nx] = size(X);%nx = nx/2;
[~,ny] = size(Y);%ny = ny/2;

d = d/2; %dimension ambient space

%------------------%
%-Cost on position-%
%------------------%

% C(x,y)=1/2*|x-y|^2
nablaC1 = @(x,y)repmat(x,[1 1 size(y,2)]) - ...
    repmat(reshape(y,[size(y,1) 1 size(y,2)]),[1 size(x,2)]);
C1 = @(x,y)squeeze( sum(nablaC1(x,y).^2)/2 );


%---------------------%
%-Cost on orientation-%
%---------------------%

normalsX = X(d+1:2*d,:)';
normalsY = Y(d+1:2*d,:)';

% Compute unit normals
norm_normalsX = sqrt(sum(normalsX .^2,2));
norm_normalsY = sqrt(sum(normalsY .^2,2));

unit_normalsX = normalsX ./  repmat(norm_normalsX,1,size(normalsX,2));
unit_normalsY = normalsY ./  repmat(norm_normalsY,1,size(normalsY,2));

prs_unit_norm = zeros(nx,ny);
for l=1:d
    prs_unit_norm = prs_unit_norm + (repmat(unit_normalsX(:,l),1,ny).*repmat(unit_normalsY(:,l)',nx,1));
end


% unoriented :
%C2 =  (2 - 2 * prs_unit_norm .^2);

% oriented :
C2 =  (1 - prs_unit_norm) ;

%--------%
%-Result-%
%--------%

% canonical metric (c1 + c2)
res = weight(1) * C1(X(1:d,:) ,Y(1:d,:)  ) + weight(2) * C2;

% Other pseudo metric tricks... (c1 + c1 * c2)
%res =  C1(X(1:d,:) ,Y(1:d,:) ) .* (1 + weight(2) * C2) ;

end
