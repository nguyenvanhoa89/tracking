function [m, P] = ekf_predict (m,P,f,Q,dt)
% In:
%   m - Nx1 mean state estimate of previous step
%   P - NxN state covariance of previous step
%   F - Transition matrix of discrete model (optional, default identity)
%   Q - Process noise of discrete model     (optional, default zero)
%   dt - 1 cycle
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
  if isa(f,'function_handle')
      f1 = @(m)f(m,dt);
      [m1,F]=jaccsd(f1,m); 
%       f2 = @(m1)f(m1,dt);
%       [~,W] = jaccsd(f2,m1); 
      P = F * P * F' + Q ;
      m=m1;
  else
      m = f * m;
      P = f * P * f' +Q ;
  end
   

end
