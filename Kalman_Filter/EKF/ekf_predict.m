function [m, P] = ekf_predict (m,P,f,Q)
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
    
  
  %
  % Perform prediction
  %
   [m,F]=jaccsd(f,m); 
   P = F * P * F' + Q;

end
