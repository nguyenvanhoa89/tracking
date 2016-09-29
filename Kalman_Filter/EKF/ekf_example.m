% Modified by Hoa V. Nguyen to demonstrate the Kalman Filter in a simple
% example: tracking a vehicle with constant velocity moving in 2D space. 
% Date: September 21st 2016
% Copyright (C) 2007 Jouni Hartikainen
function ekf_example
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
n = 500;
x = 1:n;
Y = zeros(size(R,1),n);
X_r = zeros(size(Q,1),n);
Q_r = mvnrnd(zeros(1,size(Q,1)),Q,n)';
R_r = mvnrnd(zeros(1,size(R,1)),R,n)';
for i = 2:n
   X_r(:,i) = feval(f,X_r(:,i-1)) + Q_r(:,i);
end
% Generate the measurements.
for i = 1:n
   Y_r(:,i) = feval(h,X_r(:,i)) + R_r(:,i);
end
clf; clc;
disp(' ');
fprintf('Filtering with KF...');
fprintf('Press any key to see the result');

plot(x,X_r(1,:),'.',x,Y_r);
legend('Real trajectory', 'Measurements');
title('Position');
pause 
% Initial guesses for the state mean and covariance.
m = zeros(size(Q,1),1);
P = q * eye(size(Q,1));

%% Space for the estimates.
MM = zeros(size(m,1), n);
PP = zeros(size(m,1), size(m,1), n);
% Filtering steps.

for i = 1:size(Y_r,2)
   [m,P] = ekf_predict(m,P,f,Q);
   [m,P] = ekf_update(m,P,Y_r(:,i),h,R);
   MM(:,i) = m;
   PP(:,:,i) = P;
end
plot(x,X_r(1,:),'.',x,MM(1,:));
legend('Real trajectory', 'Filtered');
title('Position estimation with Extended Kalman filter.');
xlabel('x');
ylabel('y');

pause

subplot(1,2,1);
plot(x,X_r(1,:),'.',x,MM(1,:));
legend('Real trajectory', 'Filtered');
title('Position estimation with Extended Kalman filter.');
xlabel('x');
ylabel('y');
subplot(1,2,2);
plot(x,X_r(2,:),'.',x,MM(2,:));
legend('Real velocity', 'Filtered');
title('Velocity estimation with Extended Kalman filter.');
xlabel('x^.');
ylabel('y^.');


end