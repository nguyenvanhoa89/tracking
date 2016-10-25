function [m,P,K,S,Lamda] = kf_update (m,P,z,H,R)

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
%   Lamda - Predictive probability (likelihood) of measurement.
 
  %
  % Perform update
  %
  if nargin < 5
    error('Too few arguments');
  end
  IM = H*m; % Predicted Measurement
  S = R + H * P * H';
  K = P * H' / S;
  m = m + K * (z - H * m);
  P = P - K * H * P ;
  
  if nargout > 4
    try 
        Lamda = mvnpdf(z,IM,(S + S.')/2);
    catch ex
        Lamda = mvnpdf(z,IM,R);
    end
    
  end
end


 

