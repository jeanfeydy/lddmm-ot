function Xall = generate_fiber(X0,l,curvature,n,v,noise)
% generate n curves of length l from the starting points X0
% with the given mean curvature and in the global direction v
% (parametres code en dur : nb_gaussienne et sigma
% qui influencent la nature du bruit)

if 0 % script
    l=15; n=20; X0=zeros(3,10); v=[1;1;2];
    curvature=1/l; noise=.5;
    Xall=generate_fiber(X0,l,curvature,n,v,noise);
    figure; hold on;
    for i=1:size(X0,2)
        plot3(Xall(1,:,i),Xall(2,:,i),Xall(3,:,i));
    end
    hold off;
    
    X0=rand(3,10); % source unique juste pour visualiser
    Xall=generate_fiber(X0,l,curvature,n,v,noise);
    figure; hold on;
    for i=1:size(X0,2)
        plot3(Xall(1,:,i),Xall(2,:,i),Xall(3,:,i));
    end
    hold off;
    
    l =0.7;noise=.2; n=50;
    X0=repmat([0;-.2;0],1,n)+ randn(3,n)*.01; curvature=0.5/l; v=[.1;.1;2];
    Xall=generate_fiber(X0,l,curvature,n,v,noise);
    % Xall = Xall/10;
    figure(1); hold on;
    trisurf(t,p(:,1),p(:,2),p(:,3))
    for i=1:size(X0,2)
        plot3(Xall(1,:,i),Xall(2,:,i),Xall(3,:,i));
    end
    hold off;
    axis equal
end

d=size(X0,1);
if d~=size(v,1)
    disp('Dimension of the direction vector does not match');
end

r=1/(curvature+eps);
theta=l/r;

th=3*pi/2+linspace(0,theta,n);
X=zeros(3,n);
X(1,:)=r*cos(th);
X(2,:)=r*sin(th)+r;

% Alignment on the direction v
v=v/norm(v,2); ref=zeros(d,1);ref(1)=1;
R=vrrotvec2mat(vrrotvec(ref,v));
X=R*X;
    
Xall=zeros(d,n,size(X0,2));
% Generate the variations :
for i=1:size(X0,2)
    nb_gaussienne=ceil(n/3); % parametre code en dur !
    sigma=(n/5)^2;
    noise_scale=max(noise,noise*max(max((dist(X0)))));
    fX=genereCourbe(n,noise_scale,nb_gaussienne,sigma);
    fY=genereCourbe(n,noise_scale,nb_gaussienne,sigma);
    fZ=genereCourbe(n,noise_scale,nb_gaussienne,sigma);
    fX=log(linspace(1,exp(1),n)).*fX; % cancel noise at X0
    fY=log(linspace(1,exp(1),n)).*fY; % cancel noise at X0
    fZ=log(linspace(1,exp(1),n)).*fZ; % cancel noise at X0
    Xall(:,:,i)=repmat(X0(:,i),1,n)+X+[fX;fY;fZ];
end

if d==2
    Xall(3,:,:)=[];
end
end


function fX = genereCourbe(n,a,ng,sigma)
% generate a curve of n points
% d'amplitude a, centr�e en 0

g=@(x,c,sigma)exp(-(x-c)^2/sigma);

c=linspace(0,1,ng)'+(2*rand(ng,1)-.5*ones(ng,1))/ng; % centres des gaussiennes
h=2*rand(ng,1)-ones(ng,1); % amplitudes et signes
X=linspace(0,1,n);
fX=zeros(size(X));
for i=1:ng
    fX=fX+arrayfun(@(x)h(i)*g(x,c(i),sigma),X);
end
fX=a*fX/(max(fX)-min(fX));
fX=fX-mean(fX);
end
