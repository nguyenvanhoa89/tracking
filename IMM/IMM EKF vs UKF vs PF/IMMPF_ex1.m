%% clear memory, screen, and close all figures
tic;
clear, clc, close all;
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
%% Discretization of the continous-time system.
[F{1},Q{1}] = lti_disc(f{1},L{1},Qc{1},dt);
F{2} = @c_turn;
sys{1} = @(k, x, uk) F{1}*x+uk;
sys{2} = @(k, x, uk) F{2}(x,dt)+uk;
%% Index
ind{1} = [1 2 3 4]';
ind{2} = [1 2 3 4 5]';
%% Process equation x[k] = sys(k, x[k-1], u[k]);
nx = 5;  % number of states
% dt = 0.1;
alpha = 0.9;
q = 50;
t1 = 5000; 
t2 = 500;
%% Discrete Pi Initial Value 
% p_transitional = [1-1/5000 1/5000; 1/500 1-1/500];

p_transitional = [0.95 0.05;
                  0.05 0.95];
S = [1 2];
% r0 = S(1);
% sys = @(k, xkm1, uk) xkm1/2 + 25*xkm1/(1+xkm1^2) + 8*cos(1.2*k) + uk; % (returns column vector)
% sys{1} = @(k, x, uk) [x(1)+ x(2)*dt;x(2); 0]+uk; % (returns column vector)
% sys{2} = @(k, x, uk) [x(1)+ x(2)*dt + x(3) * dt^2/2; x(2) + x(3) * dt; x(3) * alpha]+uk ; % (returns column vector)

%% Observation equation y[k] = obs(k, x[k], v[k]);
ny = 2;                                           % number of observations
% obs = @(k, xk, vk) xk(1)^2/20 + vk;                  % (returns column vector)
obs = @(k, x, vk) [x(1);x(2)]+ vk ;  
% obs = @(k, x, vk) x(1)+ vk ;                  % (returns column vector)

%% PDF of process noise and noise generator function
% sigma_u = q * eye(nx);
sigma_u{1} = Q{1};
sigma_u{2} = Q{2};

for i=1:length(S)
    nu = size(sigma_u{i},2); % size of the vector of process noise
    p_sys_noise{i}   = @(u) mvnpdf(u, zeros(1,nu), sigma_u{i});
    gen_sys_noise{i} = @(u) mvnrnd(zeros(1,nu),sigma_u{i},1)';  
    gen_x0{i} = @(x) mvnrnd(zeros(1,nu),sigma_u{i},1)';               % sample from p_x0 (returns column vector)
end
       % sample from p_sys_noise (returns column vector)
%% PDF of observation noise and noise generator function
% sigma_v = q^2;

sigma_v = diag([0.05 0.05]);    
nv =  size(sigma_v,1);  % size of the vector of observation noise
p_obs_noise   = @(v) mvnpdf(v, zeros(1,nv)', sigma_v);
gen_obs_noise = @(v) mvnrnd(zeros(1,nv),sigma_v,1)';         % sample from p_obs_noise (returns column vector)

%% Initial PDF
% p_x0 = @(x) normpdf(x, 0,sqrt(10));             % initial pdf


%% Transition prior PDF p(x[k] | x[k-1])
% (under the suposition of additive process noise)
% p_xk_given_xkm1 = @(k, xk, xkm1) p_sys_noise(xk - sys(k, xkm1, 0));

%% Observation likelihood PDF p(y[k] | x[k])
% (under the suposition of additive process noise)
p_yk_given_xk = @(k, yk, xk) p_obs_noise(yk - obs(k, xk, 0));


%% Number of time steps
T = 200;
Ns = 10000;
%% Separate memory space
x = zeros(nx,T);  y = zeros(ny,T);
u = zeros(nu,T);  v = zeros(nv,T);
%% Simulate system
xh0 =  zeros(nx,1);                                 % initial state
u(:,1) = zeros(nu,1);                               % initial process noise
v(:,1) = gen_obs_noise(sigma_v);          % initial observation noise
x(:,1) = [0 0 1 0 0]';
x(5,40) = 1;
x(5,110) = -1;
y(:,1) = obs(1, xh0, v(:,1));
mstate = ones(1,T);
for k = 2:T
   % here we are basically sampling from p_xk_given_xkm1 and from p_yk_given_xk
   

   if k <=40 || (k>= 91 && k <= 110) || (k>= 161 && k <= 200) 
       mstate(k) = 1;
       u(ind{1},k) = gen_sys_noise{1}() ;              % simulate process noise
       v(:,k) = gen_obs_noise();              % simulate observation noise
       x(ind{1},k) = sys{1}(k, x(ind{1},k-1), 0);     % simulate state
%        x(:,k) = sys{1}(k, x(:,k-1), u(:,k));     % simulate state
   else
       mstate(k) = 2;
       u(:,k) = gen_sys_noise{2}() ;              % simulate process noise
       v(:,k) = gen_obs_noise();              % simulate observation noise
       x(:,k) = sys{2}(k, x(:,k-1), 0);     % simulate state
%        x(:,k) = sys{2}(k, x(:,k-1), u(:,k));     % simulate state
   end
   
%    y(:,k) = obs(k, x(:,k),   0);     % simulate observation
   y(:,k) = obs(k, x(:,k),   v(:,k));     % simulate observation
end
%% Plot Measurement vs True trajectory
h = plot(y(1,:),y(2,:),'ko',...
         x(1,:),x(2,:),'g-');
legend('Measurement',...
       'True trajectory');
xlabel('x');
ylabel('y');
set(h,'markersize',2);
set(h,'linewidth',1.5);
set(gca,'FontSize',10);

pause 
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

h = plot(y(1,:),y(2,:),'ko',...
         x(1,:),x(2,:),'g-',...         
         xh(1,:),xh(2,:),'r-');
legend('Measurement',...
       'True trajectory',...
       'Filtered');
title('Estimates produced by IMMPF-filter.')
set(h,'markersize',2);
set(h,'linewidth',1.5);
set(gca,'FontSize',10);
fprintf('Press any key to see the NRMS result\n');
pause;
%% NRMSE
% IMMPF
NRMSE_PF1 = sqrt(mean((x(1,:)-xh(1,:)).^2))/(max(x(1,:))-min(x(1,:)))*1e2;
NRMSE_PF2= sqrt(mean((x(2,:)-xh(2,:)).^2))/(max(x(2,:))-min(x(2,:)))*1e2;
NRMSE_PF = 1/2*(NRMSE_PF1 + NRMSE_PF2);
fprintf('NRMSE of IMMPF              :%5.2f%%\n',NRMSE_PF);
%Original
NRMSE_ORG1 = sqrt(mean((x(1,:)-y(1,:)).^2))/(max(x(1,:))-min(x(1,:)))*1e2;
NRMSE_ORG2 = sqrt(mean((x(2,:)-y(2,:)).^2))/(max(x(2,:))-min(x(2,:)))*1e2;
NRMSE_ORG = 1/2*(NRMSE_ORG1 + NRMSE_ORG2);
fprintf('NRMSE of Original        :%5.2f%%\n',NRMSE_ORG);
NRMS1 = zeros(1,T);
NRMS2 = zeros(1,T);
NRMS = zeros(1,T);

for k=1:T
    NRMS1(k) = sqrt(mean((x(1,1:k)-xh(1,1:k)).^2))/(max(x(1,:))-min(x(1,:)))*1e2;
    NRMS2(k) = sqrt(mean((x(2,1:k)-xh(2,1:k)).^2))/(max(x(1,:))-min(x(1,:)))*1e2;
    NRMS(k) = 1/2*(NRMS1(k)  + NRMS2(k) );
end
% IMM Model probabilities 
MU = ones(1,T);
for k=1:T
    MU(k) = mean(pf.r(:,k));
end
h = plot(1:T,2-mstate,'g--',...
         1:T,2-MU','r-');         
legend('True',...
       'Filtered');
title('Probability of model 1');
ylim([-0.1,1.1]);
set(h,'markersize',2);
set(h,'linewidth',1.5);
set(gca,'FontSize',10);
pause
plot(1:T,NRMS);
title('NRMS trend');
xlabel('time');
ylabel('NRMS(%)');
pause


%}
return;
