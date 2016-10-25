function [m,P,K,S, Lamda] = ukf_update (m,P,z,h,R, X_hat, X_dev,dt)
% In:
%   m - Nx1 mean state estimate after prediction step
%   P - NxN state covariance after prediction step
%   Z - Dx1 measurement vector.
%   H - Measurement matrix.
%   R - Measurement noise covariance.
%   X_hat - transformed sampling points
%   X_dev - transformed deviations
%   dt - 1 cycle 
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
    nx= size(m,1);
    nz = size(z,1);
    kappa = 3 - nx;
    W=[kappa/(kappa+nx) 0.5/(kappa+nx)+zeros(1,2*nx)]; 
    
   [z_hat,~,P_zz,Z_dev]=ut(h,X_hat,W,nz,R,dt);
   P_xz = X_dev *  diag(W) * Z_dev';
   S = R + P_zz;
   K = P_xz / S;
   m = m + K *(z - z_hat);
   P = P -  P_xz * K';
   if nargout > 4
    try 
        Lamda = mvnpdf(z,z_hat,(S + S.')/2);
    catch ex
        Lamda = mvnpdf(z,z_hat,R);
    end
    
  end
end
