% Modified by Hoa V. Nguyen to demonstrate the Kalman Filter in a simple
% example: tracking a vehicle with constant velocity moving in 2D space. 
% Date: September 21st 2016
% Copyright (C) 2007 Jouni Hartikainen
function kf_example
% Stepsize
dt = 1;

% Process noise variance
q = 0.1;


% Discretization of the continous-time system.
F = [diag([1 1]) diag([dt dt]);
     diag([0 0]) diag([1 1])];
Q = q^2 * [dt^4/4*diag([1 1]) dt^3/2 * diag([1 1]);
           dt^3/2*diag([1 1]) dt^2* diag([1 1])  ];
% Measurement model.
H = [1 0 0 0 ;
     0 1 0 0 ];

% Variance in the measurements.
r1 = 10;
R = diag([r1 r1]);

% Generate the data.
n = 200;
Y = zeros(size(H,1),n);
X_r = zeros(size(F,1),n);
X_r(:,1) = [0 0 0 0 ]';
Q_r = mvnrnd(zeros(1,size(F,1)),Q,n)';
R_r = mvnrnd(zeros(1,size(H,1)),R,n)';
for i = 2:n
   X_r(:,i) = F*X_r(:,i-1) + Q_r(:,i);
end

% Generate the measurements.
for i = 1:n
   Y(:,i) = H*X_r(:,i) + R_r(:,i);
end
clf; clc;
disp(' ');
fprintf('Filtering with KF...');
fprintf('Press any key to see the result');

plot(X_r(1,:),X_r(2,:),Y(1,:),Y(2,:),'.',X_r(1,1),...
     X_r(2,1),'ro','MarkerSize',12);
legend('Real trajectory', 'Measurements');
title('Position');
pause 
% Initial guesses for the state mean and covariance.
m = zeros(size(F,1),1);
P = q * eye(size(F,1));

%% Space for the estimates.
MM = zeros(size(m,1), n);
PP = zeros(size(m,1), size(m,1), n);
% Filtering steps.
for i = 1:size(Y,2)
   [m,P] = kf_predict(m,P,F,Q);
   [m,P] = kf_update(m,P,Y(:,i),H,R);
   MM(:,i) = m;
   PP(:,:,i) = P;
end


subplot(1,2,1);
plot(X_r(1,:), X_r(2,:),'--', MM(1,:), MM(2,:),X_r(1,1),X_r(2,1),...
     'o','MarkerSize',12)
legend('Real trajectory', 'Filtered');
title('Position estimation with Kalman filter.');
xlabel('x');
ylabel('y');
subplot(1,2,2);
plot(X_r(3,:), X_r(4,:),'--', MM(3,:), MM(4,:),X_r(3,1),...
     X_r(4,1),'ro','MarkerSize',12);
legend('Real velocity', 'Filtered');
title('Velocity estimation with Kalman filter.');
xlabel('x^.');
ylabel('y^.');
end
