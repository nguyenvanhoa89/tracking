
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
nx = fdims;  % number of states %PF
hdims = 2;
ny = hdims; %PF
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
obs = @(k, x, vk) [x(1);x(2)]+ vk ;  
%% Variance in the measurements.
r{1} = 0.05;
R{1} = diag([r{1} r{1}]);

r{2} = 0.05;
R{2} = diag([r{2} r{2}]);    


%% Discretization of the continous-time system.
[F{1},Q{1}] = lti_disc(f{1},L{1},Qc{1},dt);
F{2} = @c_turn;
sys{1} = @(k, x, uk) F{1}*x+uk; %PF
sys{2} = @(k, x, uk) F{2}(x,dt)+uk; %PF
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
p_transitional = p_ij; %PF
S = [1 2];
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
sigma_u{1} = Q{1}; %PF
sigma_u{2} = Q{2}; %PF
for i=1:nmodels
    nu = size(Q{i},2); % size of the vector of process noise
    p_sys_noise{i}   = @(u) mvnpdf(u, zeros(1,nu), Q{i});
    gen_sys_noise{i} = @(u) mvnrnd(zeros(1,size(Q{i},1)),Q{i},1)';   
    gen_obs_noise{i} = @(v) mvnrnd(zeros(1,size(R{i},1)),R{i},1)'; 
    gen_x0{i} = @(x) mvnrnd(zeros(1,nu),sigma_u{i},1)';    
end

sigma_v = diag([0.05 0.05]);    
nv =  size(sigma_v,1);  % size of the vector of observation noise
p_obs_noise   = @(v) mvnpdf(v, zeros(1,nv)', sigma_v);
% gen_obs_noise = @(v) mvnrnd(zeros(1,nv),sigma_v,1)';         % sample from p_obs_noise (returns column vector)


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
    print('-dpng','IMMEKFUKFPF_measurement.png');
end
fprintf('Filtering with IMMEKFUKFPF...');
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



%IMMPF
p_yk_given_xk = @(k, yk, xk) p_obs_noise(yk - obs(k, xk, 0));
%% Number of time steps
T = n;
Ns = 10000;
%% Separate memory space
x = X_r;y = Y;
u = zeros(nu,T);  v = zeros(nv,T);
%% Simulate system
xh0 =  zeros(nx,1);                                 % initial state
u(:,1) = zeros(nu,1);                               % initial process noise
v(:,1) = gen_obs_noise{1}(sigma_v);          % initial observation noise
% x(:,1) = [0 0 1 0 0]';
% x(5,40) = 1;
% x(5,110) = -1;
% y(:,1) = obs(1, xh0, v(:,1));

%% Separate memory
xh = zeros(nx, T); xh(:,1) = xh0;
yh = zeros(ny, T); yh(:,1) = obs(1, xh0, 0);

pf.k               = 1;                   % initial iteration number
pf.Ns              = Ns;                 % number of particles
pf.w               = zeros(pf.Ns, T);     % weights
pf.particles       = zeros(nx, pf.Ns, T); % particles
pf.r               = ones(pf.Ns,T);       % R discrete variable initialization
pf.p_yk_given_xk   = p_yk_given_xk;       % function of the observation likelihood PDF p(y[k] | x[k])

%pf.p_x0 = p_x0;                          % initial prior PDF p(x[0])
%pf.p_xk_given_ xkm1 = p_xk_given_xkm1;   % transition prior PDF p(x[k] | x[k-1])
for i=1:length(S)
pf.ind{i}                = ind{i};
pf.gen_x0{i}          = gen_x0{i};              % function for sampling from initial pdf p_x0
pf.gen_sys_noise{i}   = gen_sys_noise{i};       % function for generating system noise
end
%% Estimate state
for k = 2:T
   fprintf('Iteration = %d/%d\n',k,T);
   % state estimation
   pf.k = k;
   
   [xh(:,k), pf] = IMMPF1(sys, y(:,k), pf,'multinomial_resampling',p_transitional);
 
   % filtered observation
   yh(:,k) = obs(k, xh(:,k), 0);
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

%IMMPF
NRMSE_PF1 = sqrt(mean((x(1,:)-xh(1,:)).^2))/(max(x(1,:))-min(x(1,:)))*1e2;
NRMSE_PF2= sqrt(mean((x(2,:)-xh(2,:)).^2))/(max(x(2,:))-min(x(2,:)))*1e2;
NRMSE_PF = 1/2*(NRMSE_PF1 + NRMSE_PF2);
fprintf('NRMSE of IMMPF           :%5.2f%%\n',NRMSE_PF);

%Original
NRMSE_ORG1 = sqrt(mean((X_r(1,:)-Y(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
NRMSE_ORG2 = sqrt(mean((X_r(2,:)-Y(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
NRMSE_ORG = 1/2*(NRMSE_ORG1 + NRMSE_ORG2);
fprintf('NRMSE of Original        :%5.2f%%\n',NRMSE_ORG);


%% Plot
h = plot(Y(1,:),Y(2,:),'ko',...
         X_r(1,:),X_r(2,:),'g-',...         
         MM1(1,:),MM1(2,:),'r-',...         
         MM2(1,:),MM2(2,:),'b-',...         
         xh(1,:),xh(2,:),'c-');
legend('Measurement',...
       'True trajectory',...
       'EKF Filtered',...
       'UKF Filtered',...
       'PF Filtered');
title('Estimates produced by IMM-filter.')
set(h,'markersize',2);
set(h,'linewidth',1.5);
set(gca,'FontSize',10);
if save_figures
    print('-dpng','IMMEKFUKFPF_estimate_position.png');
end
disp('Press any key to see state estimate result');
pause
% IMM Model probabilities 
MU3 = ones(1,T);
for k=1:T
    MU3(k) = 2-mean(pf.r(:,k));
end
h = plot(1:n,2-mstate,'g--',...
         1:n,MU1(1,:)','r-',...
         1:n,MU2(1,:)','b-',...
         1:n,MU3,'c-');         
legend('True',...
       'EKF Filtered',...
       'UKF Filtered',...
       'PF Filtered');
title('Probability of model 1');
ylim([-0.1,1.1]);
set(h,'markersize',2);
set(h,'linewidth',1.5);
set(gca,'FontSize',10);
if save_figures
    print('-dpng','IMMEKFUKFPF_model_prob.png');
end

%}
