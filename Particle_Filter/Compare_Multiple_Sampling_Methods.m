%% clear memory, screen, and close all figures
clear, clc, close all;
tic;
%% Process equation x[k] = sys(k, x[k-1], u[k]);
nx = 2;  % number of states
dt = 0.01;
q = 0.1;
g = 9.8;
% sys = @(k, xkm1, uk) xkm1/2 + 25*xkm1/(1+xkm1^2) + 8*cos(1.2*k) + uk; % (returns column vector)
sys = @(k, x, uk) [x(1)+ x(2)*dt;x(2)-g*sin(x(1)) * dt]+uk; % (returns column vector)

%% Observation equation y[k] = obs(k, x[k], v[k]);
ny = 1;                                           % number of observations
% obs = @(k, xk, vk) xk(1)^2/20 + vk;                  % (returns column vector)
obs = @(k, x, vk) sin(x(1)) + vk ;                  % (returns column vector)

%% PDF of process noise and noise generator function
sigma_u = q^2 * [dt^3/3 dt^2/2;
           dt^2/2 dt  ];
nu = size(sigma_u,1); % size of the vector of process noise
p_sys_noise   = @(u) mvnpdf(u, zeros(1,nu), sigma_u);
gen_sys_noise = @(u) mvnrnd(zeros(1,nu),sigma_u,1)';         % sample from p_sys_noise (returns column vector)
%% PDF of observation noise and noise generator function
sigma_v = 0.01^2;
nv =  size(sigma_v,1);  % size of the vector of observation noise
p_obs_noise   = @(v) mvnpdf(v, zeros(1,nv), sigma_v);
gen_obs_noise = @(v) mvnrnd(zeros(1,nv),sigma_v,1)';         % sample from p_obs_noise (returns column vector)

%% Initial PDF
% p_x0 = @(x) normpdf(x, 0,sqrt(10));             % initial pdf
gen_x0 = @(x) mvnrnd(zeros(1,nu),sigma_u,1)';               % sample from p_x0 (returns column vector)

%% Transition prior PDF p(x[k] | x[k-1])
% (under the suposition of additive process noise)
% p_xk_given_xkm1 = @(k, xk, xkm1) p_sys_noise(xk - sys(k, xkm1, 0));

%% Observation likelihood PDF p(y[k] | x[k])
% (under the suposition of additive process noise)
p_yk_given_xk = @(k, yk, xk) p_obs_noise(yk - obs(k, xk, 0));

%% Number of time steps
T = 500;

%% Separate memory space
x = zeros(nx,T);  y = zeros(ny,T);
u = zeros(nu,T);  v = zeros(nv,T);

%% Simulate system
xh0 = zeros(nu,1);                                  % initial state
u(:,1) = zeros(nu,1);                               % initial process noise
v(:,1) = gen_obs_noise(sigma_v);          % initial observation noise
x(:,1) = xh0;
y(:,1) = obs(1, xh0, v(:,1));
for k = 2:T
   % here we are basically sampling from p_xk_given_xkm1 and from p_yk_given_xk
   u(:,k) = gen_sys_noise();              % simulate process noise
   v(:,k) = gen_obs_noise();              % simulate observation noise
   x(:,k) = sys(k, x(:,k-1), u(:,k));     % simulate state
   y(:,k) = obs(k, x(:,k),   v(:,k));     % simulate observation
end
% plot(1:T,x(1,:),'.',1:T,y(1,:));
% legend('Real trajectory', 'Measurements');
% title('Position');
% xlabel('x');
% ylabel('y');
% fprintf('Filtering with PF...');
% fprintf('Press any key to see the result');
% pause 
%% Separate memory
xh = zeros(nx, T); xh(:,1) = xh0;
yh = zeros(ny, T); yh(:,1) = obs(1, xh0, 0);

xh_a = zeros(nx, T); xh_a(:,1) = xh0;
yh_a = zeros(ny, T); yh_a(:,1) = obs(1, xh0, 0);

pf.k               = 1;                   % initial iteration number
pf.Ns              = 1000;                 % number of particles
pf.w               = zeros(pf.Ns, T);     % weights
pf.particles       = zeros(nx, pf.Ns, T); % particles
pf.gen_x0          = gen_x0;              % function for sampling from initial pdf p_x0
pf.p_yk_given_xk   = p_yk_given_xk;       % function of the observation likelihood PDF p(y[k] | x[k])
pf.gen_sys_noise   = gen_sys_noise;       % function for generating system noise
%pf.p_x0 = p_x0;                          % initial prior PDF p(x[0])
%pf.p_xk_given_ xkm1 = p_xk_given_xkm1;   % transition prior PDF p(x[k] | x[k-1])

pf_a = pf;
Strategy = {'multinomial_resampling', 'systematic_resampling', 'soft_systematic_resampling', 'SIR', 'RPF','APF'};
NRMSE_Results = zeros(size(Strategy,2),1);
Execution_Time = zeros(size(Strategy,2),1);
for s = 1:size(Strategy,2)
    tStart = tic;  % TIC, pair 2  
    fprintf('Current Stategy %s',Strategy{s} );
    pf_a = pf;
    xh_a = zeros(nx, T); xh_a(:,1) = xh0;
    yh_a = zeros(ny, T); yh_a(:,1) = obs(1, xh0, 0);
    for k = 2:T
       fprintf('Iteration = %d/%d\n',k,T);
       pf_a.k = k;
       if  strcmp(Strategy{s} ,'APF')
           [xh_a(:,k), pf_a] = APF(sys, y(:,k), pf_a, 'multinomial_resampling');
       else
           [xh_a(:,k), pf_a] = particle_filter(sys, y(:,k), pf_a, Strategy{s});
       end
       yh_a(:,k) = obs(k, xh_a(:,k), 0);
    end
    NRMSE_Results(s) = sqrt(mean((x(1,:)-xh_a(1,:)).^2))/(max(x(1,:))-min(x(1,:)))*1e2;
    Execution_Time(s) = toc(tStart); 
end

for s = 1:size(Strategy,2)
fprintf('NRMSE of Mean   :%5.2f%%, %s\n',NRMSE_Results(s),Strategy{s});
end
NRMSE_ORG = sqrt(mean((x(1,:)-y(1,:)).^2))/(max(x(1,:))-min(x(1,:)))*1e2;
fprintf('NRMSE of Mean   :%5.2f%%, Original\n',NRMSE_ORG);

for s = 1:size(Strategy,2)
fprintf('Execution Time  :%5.2f s, %s\n',Execution_Time(s),Strategy{s});
end
%{


%% Estimate state
%PF
tic;
for k = 2:T
   fprintf('Iteration = %d/%d\n',k,T);
   
   % state estimation
   pf.k = k;   
   [xh(:,k), pf] = particle_filter(sys, y(:,k), pf, 'RPF');
   yh(:,k) = obs(k, xh(:,k), 0);
end
time_1 = toc;

   %APF
for k = 2:T
   fprintf('Iteration = %d/%d\n',k,T);
   pf_a.k = k;
   [xh_a(:,k), pf_a] = APF(sys, y(:,k), pf_a, 'SIR');
   yh_a(:,k) = obs(k, xh_a(:,k), 0);
end
time_2 = toc - time_1;

plot(1:T,x(1,:),'.',1:T,xh(1,:),'x',1:T,xh_a(1,:));
legend('Real trajectory', 'RPF Filtered', 'APF Filtered');
title('Position estimation with Particle filter.');
xlabel('x');
ylabel('y');
MSE_1 = mean((x(1,:)-xh(1,:)).^2)/mean((x(1,:)).^2)*1e2;
MSE_2 = mean((x(1,:)-xh_a(1,:)).^2)/mean((x(1,:)).^2)*1e2;
% MSE_KF2 = mean((x(2,:)-xh(2,:)).^2)*1e5;
% MSE_KF = 1/2*(MSE_KF1 + MSE_KF2);
MSE_ORG = mean((x(1,:)-y(1,:)).^2)/mean((x(1,:)).^2)*1e2;
fprintf('MSE of Mean RPF          :%5.2f\n',MSE_1);
fprintf('MSE of Mean APF         :%5.2f\n',MSE_2);
% fprintf('MSE of Mean 2                               :%5.2f\n',MSE_KF2);
% fprintf('MSE of Mean all                               :%5.2f\n',MSE_KF);
fprintf('MSE of Mean Original    :%5.2f\n',MSE_ORG);
fprintf('SIR Execution Time       :%5.2f\n',time_1);
fprintf('APF Execution Time      :%5.2f\n',time_2);

%}

time_whole = toc;
return;