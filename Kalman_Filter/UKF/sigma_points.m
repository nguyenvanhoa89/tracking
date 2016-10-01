function X=sigma_points(m,P,c)
%Sigma points around reference point
%Inputs:
%       x: reference point
%       P: covariance
%       c: coefficient
%Output:
%       X: Sigma points

A = c*chol(P)';
Y = m(:,ones(1,numel(m)));
X = [m Y+A Y-A];
end