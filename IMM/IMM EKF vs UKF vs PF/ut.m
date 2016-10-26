function [y,Y,P,Y1]=ut(f,X,W,n,R,dt)
%Unscented Transformation
%Input:
%        f: nonlinear map
%        X: sigma points
%       W: weights for mean
%        n: numer of outputs of f
%        R: additive covariance
%   dt - 1 cycle 
%Output:
%        y: transformed mean
%        Y: transformed smapling points
%        P: transformed covariance
%       Y1: transformed deviations

L=size(X,2);
y=zeros(n,1);
Y=zeros(n,L);
for k=1:L                   
    if isa(f,'function_handle')
        Y(:,k)=f(X(:,k),dt);      
    else
        Y(:,k)=f*(X(:,k));      
    end     
    y=y+W(k)*Y(:,k);       
end
Y1=Y-y(:,ones(1,L));
P=Y1*diag(W)*Y1'+R; 
end
