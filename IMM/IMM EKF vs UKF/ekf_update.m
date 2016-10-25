function [m,P,K,S, Lamda] = ekf_update (m,P,z,h,R)
% In:
%   m - Nx1 mean state estimate after prediction step
%   P - NxN state covariance after prediction step
%   Z - Dx1 measurement vector.
%   H - Measurement matrix.
%   R - Measurementamda noise covariance.
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
  
  if isa(h,'function_handle')
    
      [z1, H] = jaccsd(h,m);
      S = R + H * P * H';
      K = P * H' / S;
      m = m + K * (z - z1);
      P = P - K * H * P ;
  else
      H = h;
      IM = H*m; % Predicted Measurement
      S = R + H * P * H';
      K = P * H' / S;
      m = m + K * (z - H * m);
      P = P - K * H * P ;
  end
      
  
  if nargout > 4
    try 
        Lamda = mvnpdf(z,IM,(S + S.')/2);
    catch ex
        Lamda = mvnpdf(z,IM,R);
    end
    
  end

end
