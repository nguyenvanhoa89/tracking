
% Modified by Hoa Nguyen - October 2016.
% EIMM_DEMO1 Coordinated turn model demonstration 
% 
% Simple demonstration for non-linear IMM using the following models:
%  1. Standard Wiener process velocity model
%  2. Coordinated turn model
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
%% Save figures
print_figures = 1;
save_figures = 1;
%% Dimensionality of the state space
fdims = 5; %[x1 x2 v1 v2 w]
hdims = 2;
nmodels = 2;

%% Stepsize
dt = 0.1;

%% Transition matrix for the continous-time velocity model.
f{1} = [0 0 1 0;
        0 0 0 1;
        0 0 0 0;
        0 0 0 0];
%% Noise effect matrix for the continous-time system.
L{1} = [0 0;
        0 0;
        1 0;
        0 1];
L{2} = [0 0 0 0 1]'; % Noise at turn rate only

%% Process noise variance
q{1} = 0.05;
Qc{1} = diag([q{1} q{1}]);
Qc{2} = 0.15;
Q{2} = L{2}*Qc{2}*L{2}'*dt;

%% Measurement models.
H{1} = [1 0 0 0;
        0 1 0 0];

H{2} = [1 0 0 0 0 ;
        0 1 0 0 0 ];
%% Variance in the measurements.
r{1} = 0.05;
R{1} = diag([r{1} r{1}]);

r{2} = 0.05;
R{2} = diag([r{2} r{2}]);    


%% Discretization of the continous-time system.
[F{1},Q{1}] = lti_disc(f{1},L{1},Qc{1},dt);
F{2} = @c_turn;
%% Index
ind{1} = [1 2 3 4]';
ind{2} = [1 2 3 4 5]';


%% Generate the data.
n = 200;

X_r = zeros(fdims,n);
Q_r = zeros(fdims,n);

Y = zeros(hdims,n);
R_r = zeros(hdims,n);

mstate = zeros(1,n);

w1 =  [0.95 0.05];

w2 =  [0.95 0.05];

p_ij = [0.95 0.05;
        0.05 0.95];

%% Forced mode transitions 
% Start with constant velocity 1 toward right
mstate(1:40) = 1;
X_r(:,1) = [0 0 1 0 0]';
% At 4s make a turn left with rate 1 
mstate(41:90) = 2;
X_r(5,40) = 1;
% At 9s move straight for 2 seconds
mstate(91:110) = 1;

% At 11s commence another turn right with rate -1
mstate(111:160) = 2;
X_r(5,110) = -1;

% At 16s move straight for 4 seconds
mstate(161:200) = 1;  
%% Get noise model

for i=1:nmodels
gen_sys_noise{i} = @(u) mvnrnd(zeros(1,size(Q{i},1)),Q{i},1)';   
gen_obs_noise{i} = @(v) mvnrnd(zeros(1,size(R{i},1)),R{i},1)'; 
end

%% Process model
    for i = 2:n
       st = mstate(i);
       Q_r(ind{st},i) = gen_sys_noise{st}();
       if isa(F{st},'function_handle')
           X_r(ind{st},i) = F{st}(X_r(ind{st},i-1),dt);% + Q_r(ind{st},i);
       else
           X_r(ind{st},i) = F{st}*X_r(ind{st},i-1);% + Q_r(ind{st},i);
       end
    end
%% Generate the measurements.
for i = 1:n
    st = mstate(i);
    R_r(:,i) = gen_obs_noise{st}();
     if isa(H{st},'function_handle')
        Y(:,i) = H{st}(X_r(ind{st},i)) + R_r(:,i);
     else
         Y(:,i) = H{st}*X_r(ind{st},i) + R_r(:,i);
     end
end

%% Plot Measurement vs True trajectory
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
    print('-dpng','IMMEKFUKF_measurement.png');
end
fprintf('Filtering with IMMEKFUKF...');
fprintf('Press any key to see the result\n');

pause 
%%{
%% Initial Values %%

% KF Model 1
KF_M = zeros(size(F{1},1),1);
KF_P = 0.1 * eye(size(F{1},1));

% IMMEKF (1)
x_ip1{1} = zeros(size(F{1},1),1);
P_ip1{1} = 0.1 * eye(size(F{1},1));
x_ip1{2} = zeros(fdims,1);
P_ip1{2} = 0.1 * eye(fdims);

% IMMUKF (2)
x_ip2{1} = zeros(size(F{1},1),1);
P_ip2{1} = 0.1 * eye(size(F{1},1));
x_ip2{2} = zeros(fdims,1);
P_ip2{2} = 0.1 * eye(fdims);

%% Space For Estimation %%

% KF Model 1 Filter
KF_MM = zeros(size(F{1},1),  n);
KF_PP = zeros(size(F{1},1), size(F{1},1), n);

% IMM
% Model-conditioned estimates of IMM EKF
MM1_i = cell(2,n);
PP1_i = cell(2,n);
% Model-conditioned estimates of IMM UKF
MM2_i = cell(2,n);
PP2_i = cell(2,n);

% Overall estimates of IMM filter
%IMMEKF
MM1 = zeros(fdims,  n);
PP1 = zeros(fdims, fdims, n);
%IMMUKF
MM2 = zeros(fdims,  n);
PP2 = zeros(fdims, fdims, n);

% IMM Model probabilities 
MU1 = zeros(2,n); %IMMEKF
MU2 = zeros(2,n); %IMMUKF

%% Filtering steps. %%
for i = 1:n
    %KF model 1
    [KF_M,KF_P] = kf_predict(KF_M,KF_P,F{1},Q{1});
    [KF_M,KF_P] = kf_update(KF_M,KF_P,Y(:,i),H{1},R{1});
    KF_MM(:,i)   = KF_M;
    KF_PP(:,:,i) = KF_P;
    %IMMEKF
    [x_p1,P_p1,c_j1] = eimm_predict(x_ip1,P_ip1,w1,p_ij,ind,fdims,F,Q,dt);
    [x_ip1,P_ip1,w1,m1,P1] = eimm_update(x_p1,P_p1,c_j1,ind,fdims,Y(:,i),H,R);
    MM1(:,i)   = m1;
    PP1(:,:,i) = P1;
    MU1(:,i)   = w1';
    MM1_i(:,i) = x_ip1';
    PP1_i(:,i) = P_ip1';
    %IMMUKF
    [x_p2,P_p2,c_j2,X_hat, X_dev] = uimm_predict(x_ip2,P_ip2,w2,p_ij,ind,fdims,F,Q,dt);
%     [x_ip2,P_ip2,w2,m2,P2] = imm_update(x_p2,P_p2,c_j2,ind,fdims,Y(:,i),H,R);
    [x_ip2,P_ip2,w2,m2,P2] = uimm_update(x_p2,P_p2,c_j2,ind,fdims,Y(:,i),H,R,X_hat, X_dev, dt);
    MM2(:,i)   = m2;
    PP2(:,:,i) = P2;
    MU2(:,i)   = w2';
    MM2_i(:,i) = x_ip2';
    PP2_i(:,i) = P_ip2';
end

%% Calculate Normalise Root Mean Square Error (NRMSE)
%KF Model 1
NRMSE_KF1_1 = sqrt(mean((X_r(1,:)-KF_MM(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
NRMSE_KF1_2 = sqrt(mean((X_r(2,:)-KF_MM(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
NRMSE_KF1 = 1/2*(NRMSE_KF1_1 + NRMSE_KF1_2);
fprintf('NRMSE of KF1             :%5.2f%%\n',NRMSE_KF1);

%IMM EKF
NRMSE_IMMEKF1 = sqrt(mean((X_r(1,:)-MM1(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
NRMSE_IMMEKF2 = sqrt(mean((X_r(2,:)-MM1(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
NRMSE_IMMEKF = 1/2*(NRMSE_IMMEKF1 + NRMSE_IMMEKF2);
fprintf('NRMSE of IMMEKF          :%5.2f%%\n',NRMSE_IMMEKF);

%IMM UKF
NRMSE_IMMUKF1 = sqrt(mean((X_r(1,:)-MM2(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
NRMSE_IMMUKF2 = sqrt(mean((X_r(2,:)-MM2(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
NRMSE_IMMUKF = 1/2*(NRMSE_IMMUKF1 + NRMSE_IMMUKF2);
fprintf('NRMSE of IMMUKF          :%5.2f%%\n',NRMSE_IMMUKF);

%Original
NRMSE_ORG1 = sqrt(mean((X_r(1,:)-Y(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
NRMSE_ORG2 = sqrt(mean((X_r(2,:)-Y(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
NRMSE_ORG = 1/2*(NRMSE_ORG1 + NRMSE_ORG2);
fprintf('NRMSE of Original        :%5.2f%%\n',NRMSE_ORG);

%% Plot
h = plot(Y(1,:),Y(2,:),'ko',...
         X_r(1,:),X_r(2,:),'g-',...         
         MM1(1,:),MM1(2,:),'r-',...         
         MM2(1,:),MM2(2,:),'b-');
legend('Measurement',...
       'True trajectory',...
       'EKF Filtered',...
       'UKF Filtered');
title('Estimates produced by IMM-filter.')
set(h,'markersize',2);
set(h,'linewidth',1.5);
set(gca,'FontSize',10);
if save_figures
    print('-dpng','IMMEKFUKF_estimate_position.png');
end
disp('Press any key to see state estimate result');
pause
h = plot(1:n,2-mstate,'g--',...
         1:n,MU1(1,:)','r-',...
         1:n,MU2(1,:)','ko-');         
legend('True',...
       'EKF Filtered',...
       'UKF Filtered');
title('Probability of model 1');
ylim([-0.1,1.1]);
set(h,'markersize',2);
set(h,'linewidth',1.5);
set(gca,'FontSize',10);
if save_figures
    print('-dpng','IMMEKFUKF_model_prob.png');
end

%}
