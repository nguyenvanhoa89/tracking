% Modified by Hoa Nguyen - October 2016.
%IMM_DEMO  Tracking a Target with Simple Manouvers demonstration
% 
% Simple demonstration for linear IMM using the following models:
%  1. Standard Wiener process velocity model
%  2. Standard Wiener process acceleration model
%
% The measurement model is linear, which gives noisy measurements 
% of target's position.
% 
% Copyright (C) 2007-2008 Jouni Hartikainen
%
% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.

clear; clc;
% Save figures
print_figures = 1;
save_figures = 1;

% Dimensionality of the state space
fdims = 6;
hdims = 2;
nmodels = 2;

% Process noise variance
q{1} = 0.01;
Qc{1} = diag([q{1} q{1}]);

q{2} = 1;
Qc{2} = diag([q{2} q{2}]);
% Variance in the measurements.
r{1} = 0.1;
R{1} = diag([r{1} r{1}]);

r{2} = 0.1;
R{2} = diag([r{2} r{2}]);

% Transition matrix for the continous-time velocity model.
f{1} = [0 0 1 0;
        0 0 0 1;
        0 0 0 0;
        0 0 0 0];
f{2} = [0 0 1 0 0 0;
        0 0 0 1 0 0;
        0 0 0 0 1 0;
        0 0 0 0 0 1;
        0 0 0 0 0 0;
        0 0 0 0 0 0];

% Noise effect matrix for the continous-time system.
L{1} = [0 0;
        0 0;
        1 0;
        0 1];

L{2} = [0 0;
        0 0;
        0 0;
        0 0;
        1 0;
        0 1];
    
% Measurement models.
H{1} = [1 0 0 0;
        0 1 0 0];

H{2} = [1 0 0 0 0 0;
        0 1 0 0 0 0];
% Stepsize
dt = 0.1;

% Discretization of the continous-time system.
for i=1:nmodels
[F{i},Q{i}] = lti_disc(f{i},L{i},Qc{i},dt);
end
% Index
ind{1} = [1 2 3 4]';
ind{2} = [1 2 3 4 5 6]';

% Generate the data.
n = 200;

X_r = zeros(fdims,n);
X_r(:,1) = zeros(size(F{2},1),1);
Q_r = zeros(fdims,n);

Y = zeros(hdims,n);
R_r = zeros(hdims,n);

mstate = zeros(1,n);

w = [0.9 0.1];

p_ij = [0.95 0.05;
        0.05 0.95];

% Forced mode transitions 
mstate(1:50) = 1;
mstate(51:70) = 2;
mstate(71:120) = 1;
mstate(121:150) = 2;
mstate(151:200) = 1;    
    
% Get noise model
for i=1:nmodels
gen_sys_noise{i} = @(u) mvnrnd(zeros(1,size(Q{i},1)),Q{i},1)';   
gen_obs_noise{i} = @(v) mvnrnd(zeros(1,size(R{i},1)),R{i},1)'; 
end

% Process model
for i = 2:n
   st = mstate(i);
   Q_r(ind{st},i) = gen_sys_noise{st}();
   X_r(ind{st},i) = F{st}*X_r(ind{st},i-1) + Q_r(ind{st},i);
end
    

% Generate the measurements.
for i = 1:n
    st = mstate(i);
    R_r(:,i) = gen_obs_noise{st}();
    Y(:,i) = H{st}*X_r(ind{st},i) + R_r(:,i);
end

h = plot(Y(1,:),Y(2,:),'ko',...
         X_r(1,:),X_r(2,:),'g-');
legend('Measurement',...
       'True trajectory');
xlabel('x');
ylabel('y');
set(h,'markersize',2);
set(h,'linewidth',1.5);
set(gca,'FontSize',10);
if save_figures
    print('-dpng','immkf_measurement.png');
end
fprintf('Filtering with IMMKF...');
fprintf('Press any key to see the result\n');

pause 


% Initial Values
m = zeros(size(F{2},1),1);
P = 0.1 * eye(6);
% KF1
M1 = zeros(size(F{1},1),1);
P1 = 0.1 * eye(4);

% KF2
M2 = zeros(size(F{2},1),1);
P2 = 0.1 * eye(6);
%%%% Space for the estimates %%%% 
% KF using model 1 
MM1 = zeros(size(F{1},1),n);
PP1 = zeros(size(F{1},1), size(F{1},1), n);

% KF using model 2
MM2 = zeros(size(F{2},1), n);
PP2 = zeros(size(F{2},1), size(F{2},1), n);

% Model-conditioned estimates of IMM
MM_i = cell(2,n);
PP_i = cell(2,n);

% Overall estimates of IMM filter
MM = zeros(size(m,1),  n);
PP = zeros(size(m,1), size(m,1), n);

% Model probabilities 
MU = zeros(2,n);

% Initialise model values - IMM
x_ip{1} = zeros(size(F{1},1),1);
P_ip{1} = 0.1 * eye(4);
x_ip{2} = zeros(size(F{2},1),1);
P_ip{2} = 0.1 * eye(6);

% Filtering steps.
for i = 1:n
    
    fprintf('%5.0f\n',i);
    % KF with model 1
    [M1,P1] = kf_predict(M1,P1,F{1},Q{1});
    [M1,P1] = kf_update(M1,P1,Y(:,i),H{1},R{1});
    MM1(:,i) = M1;
    PP1(:,:,i) = P1; 
    
    % KF with model 2
    [M2,P2] = kf_predict(M2,P2,F{2},Q{2});
    [M2,P2] = kf_update(M2,P2,Y(:,i),H{2},R{2});
    MM2(:,i) = M2;
    PP2(:,:,i) = P2;
    
    %IMM
    [x_p,P_p,c_j] = imm_predict(x_ip,P_ip,w,p_ij,ind,fdims,F,Q);
    [x_ip,P_ip,w,m,P] = imm_update(x_p,P_p,c_j,ind,fdims,Y(:,i),H,R);
    MM(:,i)   = m;
    PP(:,:,i) = P;
    MU(:,i)   = w';
    MM_i(:,i) = x_ip';
    PP_i(:,i) = P_ip';
end
%KF Model 1
NRMSE_KF1_1 = sqrt(mean((X_r(1,:)-MM1(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
NRMSE_KF1_2 = sqrt(mean((X_r(2,:)-MM1(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
NRMSE_KF1 = 1/2*(NRMSE_KF1_1 + NRMSE_KF1_2);
fprintf('NRMSE of KF1             :%5.2f%%\n',NRMSE_KF1);
%KF Model 2
NRMSE_KF2_1 = sqrt(mean((X_r(1,:)-MM2(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
NRMSE_KF2_2 = sqrt(mean((X_r(2,:)-MM2(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
NRMSE_KF2 = 1/2*(NRMSE_KF2_1 + NRMSE_KF2_2);
fprintf('NRMSE of KF2             :%5.2f%%\n',NRMSE_KF2);
%IMM
NRMSE_IMM1 = sqrt(mean((X_r(1,:)-MM(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
NRMSE_IMM2 = sqrt(mean((X_r(2,:)-MM(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
NRMSE_IMM = 1/2*(NRMSE_IMM1 + NRMSE_IMM2);
fprintf('NRMSE of IMM             :%5.2f%%\n',NRMSE_IMM);
NRMSE_ORG1 = sqrt(mean((X_r(1,:)-Y(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
NRMSE_ORG2 = sqrt(mean((X_r(2,:)-Y(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
NRMSE_ORG = 1/2*(NRMSE_ORG1 + NRMSE_ORG2);
fprintf('NRMSE of Original        :%5.2f%%\n',NRMSE_ORG);
h = plot(Y(1,:),Y(2,:),'ko',...
         X_r(1,:),X_r(2,:),'g-',...         
         MM(1,:),MM(2,:),'r-');
legend('Measurement',...
       'True trajectory',...
       'Filtered');
title('Estimates produced by IMM-filter.')
set(h,'markersize',2);
set(h,'linewidth',1.5);
set(gca,'FontSize',10);
if save_figures
    print('-dpng','immkf_estimate_position.png');
end
disp('Press any key to see state estimate result');
pause
h = plot(1:n,2-mstate,'g--',...
         1:n,MU(1,:)','r-');         
legend('True',...
       'Filtered');
title('Probability of model 1');
ylim([-0.1,1.1]);
set(h,'markersize',2);
set(h,'linewidth',1.5);
set(gca,'FontSize',10);
if save_figures
    print('-dpng','immkf_model_prob.png');
end