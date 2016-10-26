function  [m_hat,X_hat,P_hat,X_dev] = ukf_predict (m,P,f,Q,dt)
% In:
%   m - Nx1 mean state estimate of previous step
%   P - NxN state covariance of previous step
%   F - Transition matrix of discrete model (optional, default identity)
%   Q - Process noise of discrete model     (optional, default zero)
%   dt - 1 cycle 
%
% Out:
%   m_hat - Predicted state mean
%   P_hat - Predicted state covariance
%   X_hat - transformed sampling points
%   X_dev - transformed deviations

%   
%
  % Apply defaults
  %
  % Initial guesses for the state mean and covariance.
    nx= size(m,1);
    kappa = 3 - nx;
    W=[kappa/(kappa+nx) 0.5/(kappa+nx)+zeros(1,2*nx)]; 
    c= sqrt(kappa + nx);  
  
  %
  % Perform prediction
  %
    X = sigma_points(m,P,c);
    [m_hat,X_hat,P_hat,X_dev]=ut(f,X,W,nx,Q,dt);
end
