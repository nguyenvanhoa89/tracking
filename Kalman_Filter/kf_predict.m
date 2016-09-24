function [m, P] = kf_predict (m,P,F,Q)
% In:
%   m - Nx1 mean state estimate of previous step
%   P - NxN state covariance of previous step
%   F - Transition matrix of discrete model (optional, default identity)
%   Q - Process noise of discrete model     (optional, default zero)
%
% Out:
%   m - Predicted state mean
%   P - Predicted state covariance
%   
%
  % Apply defaults
  %
  if isempty(F)
    F = eye(size(m,1));
  end
  if isempty(Q)
    Q = zeros(size(m,1));
  end
  
  %
  % Perform prediction
  %
   m = F * m;
   P = F * P * F' + Q;

end
