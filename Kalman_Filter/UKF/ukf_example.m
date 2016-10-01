% Modified by Hoa V. Nguyen to demonstrate the Unscented Kalman Filter in a simple
% example: tracking a pendulum trajectories through its alpha corner value
% Date: October 1st 2016
function ukf_example
% Stepsize
dt = 0.01;

% Process noise variance
q = 0.1;

% Discretization of the continous-time system.
Q = q^2 * [dt^3/3 dt^2/2;
           dt^2/2 dt  ];
% Variance in the measurements.
r1 = 0.01;
R = r1^2;
% State function
g= 9.8;
f=@(x)[x(1)+ x(2)*dt;x(2)-g*sin(x(1)) * dt];
h=@(x)[sin(x(1))];

% Generate the data.
N = 500;
x = 1:N;
X_r = zeros(size(Q,1),N);
Q_r = mvnrnd(zeros(1,size(Q,1)),Q,N)';
R_r = mvnrnd(zeros(1,size(R,1)),R,N)';
for i = 2:N
   X_r(:,i) = feval(f,X_r(:,i-1)) + Q_r(:,i);
end
% Generate the measurements.
for i = 1:N
   Z_r(:,i) = feval(h,X_r(:,i)) + R_r(:,i);
end
clf; clc;
disp(' ');
fprintf('Filtering with KF...');
fprintf('Press any key to see the result');

plot(x,X_r(1,:),'.',x,Z_r);
legend('Real trajectory', 'Measurements');
title('Position');
pause 
% Initial guesses for the state mean and covariance.
nx= size(Q,1);
nz = size(R,1);
m = zeros(nx,1);
P = q*eye(nx);  
kappa = 3 - nx;
W=[kappa/(kappa+nx) 0.5/(kappa+nx)+zeros(1,2*nx)]; 
c= sqrt(kappa + nx);

%% Space for the estimates.
MM = zeros(nx, N);
PP = zeros(nx, nx, N);

for i = 1:size(Z_r,2)
   % Filtering steps.
   X=sigma_points(m,P,c);
   [m_hat,X_hat,P_hat,X_dev]=ut(f,X,W,nx,Q);
   % Update steps.
   [z_hat,Z_hat,P_zz,Z_dev]=ut(h,X_hat,W,nz,R);
   P_xz = X_dev *  diag(W) * Z_dev';
   S = R + P_zz;
   K = P_xz / S;
   m = m_hat + K *(Z_r(:,i) - z_hat);
   P = P_hat -  P_xz * K';
   MM(:,i) = m;
   PP(:,:,i) = P;
end
plot(x,X_r(1,:),'.',x,MM(1,:));
legend('Real trajectory', 'Filtered');
title('Position estimation with Unscented Kalman filter.');
xlabel('x');
ylabel('y');

end