function [m,P,K,S] = kf_update (m,P,z,H,R)
% In:
%   m - Nx1 mean state estimate after prediction step
%   P - NxN state covariance after prediction step
%   Z - Dx1 measurement vector.
%   H - Measurement matrix.
%   R - Measurement noise covariance.
%
% Out:
%   m  - Updated state mean
%   P  - Updated state covariance
%   K  - Computed Kalman gain
%   S  - Covariance or predictive mean of Y
 
  %
  % Perform update
  %
  S = R + H * P * H';
  K = P * H' / S;
  m = m + K * (z - H * m);
  P = P - K * H * P ;

end
