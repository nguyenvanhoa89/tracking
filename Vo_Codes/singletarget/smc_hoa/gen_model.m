function model= gen_model

% basic parameters
model.x_dim= 4;   %dimension of state vector
model.z_dim= 1;   %dimension of observation vector

% dynamical model parameters (CV model)
model.T= 1;                                     %sampling period
model.A0= [ 1 model.T; 0 1 ];                         %transition matrix                     
model.F= [ model.A0 zeros(2,2); zeros(2,2) model.A0 ];
model.B0= [ (model.T^2)/2; model.T ];
model.B= [ model.B0 zeros(2,1); zeros(2,1) model.B0 ];
model.sigma_v = 5;
model.Q= (model.sigma_v)^2* model.B*model.B';   %process noise covariance

% survival/death parameters
% N/A for single target

% birth parameters
% N/A for single target

% observation model parameters (noisy r/theta only)
% measurement transformation given by gen_observation_fn, observation matrix is N/A in non-linear case
model.D= diag( 2 );      %std for angle and range noise
model.R= model.D*model.D';              %covariance for observation noise

% detection parameters
% use compute_pD for state dependent parameterization
% use compute_qD for state dependent parameterization

% clutter parameters
model.lambda_c= 20;                             %poisson average rate of uniform clutter (per scan)
model.range_c= [ -1000 1000 ];          %uniform clutter on r/theta
% model.pdf_c= 1/prod(model.range_c(:,2)-model.range_c(:,1)); %uniform clutter density
model.pdf_c = 1/norm(model.range_c);




