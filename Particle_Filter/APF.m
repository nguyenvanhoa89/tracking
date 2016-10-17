function [xhk, pf] = APF(sys, yk, pf, resampling_strategy)
%% Auxiliary particle filter
%
% Note: when resampling is performed on each step this algorithm is called
% the Bootstrap particle filter
%
% Usage:
% [xhk, pf] = particle_filter(sys, yk, pf, resamping_strategy)
%
% Inputs:
% sys  = function handle to process equation
% yk   = observation vector at time k (column vector)
% pf   = structure with the following fields
%   .k                = iteration number
%   .Ns               = number of particles
%   .w                = weights   (Ns x T)
%   .particles        = particles (nx x Ns x T)
%   .gen_x0           = function handle of a procedure that samples from the initial pdf p_x0
%   .p_yk_given_xk    = function handle of the observation likelihood PDF p(y[k] | x[k])
%   .gen_sys_noise    = function handle of a procedure that generates system noise
% resampling_strategy = resampling strategy. Set it either to 
%                       'multinomial_resampling' or 'systematic_resampling'
%
% Outputs:
% xhk   = estimated state
% pf    = the same structure as in the input but updated at iteration k
%
% Reference:
% [1] Arulampalam et. al. (2002).  A tutorial on particle filters for 
%     online nonlinear/non-gaussian bayesian tracking. IEEE Transactions on 
%     Signal Processing. 50 (2). p 174--188

%% Programmed by:
% Diego Andres Alvarez Marin (diegotorquemada@gmail.com)
% Universidad Nacional de Colombia at Manizales, February 29, 2012

%%
k = pf.k;
if k == 1
   error('error: k must be an integer greater or equal than 2');
end

%% Initialize variables
Ns = pf.Ns;                              % number of particles
nx = size(pf.particles,1);               % number of states

wkm1 = pf.w(:, k-1);                     % weights of last iteration
if k == 2
   for i = 1:Ns                          % simulate initial particles
      pf.particles(:,i,1) = pf.gen_x0(); % at time k=1
   end   
   wkm1 = repmat(1/Ns, Ns, 1);           % all particles have the same weight
end

%%
% The importance sampling function:
% PRIOR: (this method is sensitive to outliers)   THIS IS THE ONE USED HERE
% q_xk_given_xkm1_yk = pf.p_xk_given_xkm1;

% OPTIMAL:
% q_xk_given_xkm1_yk = q_xk_given_xkm1^i_yk;
% Note this PDF can be approximated by MCMC methods: they are expensive but 
% they may be useful when non-iterative schemes fail

%% Separate memory
xkm1 = pf.particles(:,:,k-1); % extract particles from last iteration;
xk   = zeros(size(xkm1));     % = zeros(nx,Ns);
wk   = zeros(size(wkm1));     % = zeros(Ns,1);
vk   = zeros(size(wkm1));     % Parent weight
iParents  = zeros(size(wkm1));
%% Algorithm 3 of Ref [1]

   % xk(:,i) = sample_vector_from q_xk_given_xkm1_yk given xkm1(:,i) and yk
   % Using the PRIOR PDF: pf.p_xk_given_xkm1: eq 62, Ref 1.
   
   for i=1:Ns
       vk(i) = wkm1(i) * pf.p_yk_given_xk(k, yk, sys(k-1, xkm1(:,i), zeros(nx,1) ));
   end
   %% Normalize parent weight vector
   vk = vk./sum(vk);
%    iPick     = 1;
%    c         = cumsum(vk);
%    threshold = rand/Ns;
%    for i=1:Ns
%         while threshold>c(iPick) && iPick<Ns, iPick = iPick + 1; end;
%         iParents(i) = iPick;
%         threshold    = threshold + 1/Ns;                
%    end
   
   [~, ~, iParents] = resample(xk, vk, resampling_strategy);
   for i=1:Ns
        iParent    = iParents(i);
        xk(:,i) = sys(k, xkm1(:,iParent), pf.gen_sys_noise()); 
        wk(i)  = wkm1(iParent) * pf.p_yk_given_xk(k, yk, xk(:,i))/vk(iParent);
   end
   
   % Equation 48, Ref 1.
   % wk(i) = wkm1(i) * p_yk_given_xk(yk, xk(:,i))*p_xk_given_xkm1(xk(:,i), xkm1(:,i))/q_xk_given_xkm1_yk(xk(:,i), xkm1(:,i), yk);
   
   % weights (when using the PRIOR pdf): eq 63, Ref 1
%    wk(i) = wkm1(i) * pf.p_yk_given_xk(k, yk, xk(:,i));
   
   % weights (when using the OPTIMAL pdf): eq 53, Ref 1
   % wk(i) = wkm1(i) * p_yk_given_xkm1(yk, xkm1(:,i)); % we do not know this PDF


%% Normalize weight vector
wk = wk./sum(wk);

%% Calculate effective sample size: eq 48, Ref 1
Neff = 1/sum(wk.^2);

%% Resampling
% remove this condition and sample on each iteration:
% [xk, wk] = resample(xk, wk, resampling_strategy);
%if you want to implement the bootstrap particle filter
% resample_percentaje = 0.0;
% Nt = resample_percentaje*Ns;
% if Neff < Nt
%    disp('Resampling ...')
%    [xk, wk] = resample(xk, wk, resampling_strategy);
%    % {xk, wk} is an approximate discrete representation of p(x_k | y_{1:k})
% end

%% Compute estimated state
xhk = zeros(nx,1);
for i = 1:Ns;
   xhk = xhk + wk(i)*xk(:,i);
end

%% Store new weights and particles
pf.w(:,k) = wk;
pf.particles(:,:,k) = xk;

return; % bye, bye!!!

%% Resampling function
function [xk, wk, idx] = resample(xk, wk, resampling_strategy)

Ns = length(wk);  % Ns = number of particles

% wk = wk./sum(wk); % normalize weight vector (already done)

switch resampling_strategy
   case 'multinomial_resampling'
      with_replacement = true;
      idx = randsample(1:Ns, Ns, with_replacement, wk);
%{
      THIS IS EQUIVALENT TO:
      edges = min([0 cumsum(wk)'],1); % protect against accumulated round-off
      edges(end) = 1;                 % get the upper edge exact
      % this works like the inverse of the empirical distribution and returns
      % the interval where the sample is to be found
      [~, idx] = histc(sort(rand(Ns,1)), edges);
%}
   case 'systematic_resampling'
      % this is performing latin hypercube sampling on wk
      edges = min([0 cumsum(wk)'],1); % protect against accumulated round-off
      edges(end) = 1;                 % get the upper edge exact
      u1 = rand/Ns;
      % this works like the inverse of the empirical distribution and returns
      % the interval where the sample is to be found
      [~, idx] = histc(u1:1/Ns:1, edges);
   % case 'regularized_pf'      TO BE IMPLEMENTED
   % case 'stratified_sampling' TO BE IMPLEMENTED
   % case 'residual_sampling'   TO BE IMPLEMENTED
    case 'SIR'
        idx = zeros(size(wk));
        N=length(wk);
        w=cumsum(wk);
        bin=1/N*rand(1)+1/N*(0:N-1);idxx=1;
        for t=1:N
            while bin(t)>=w(idxx)
                idxx=idxx+1;
            end
        idx(t) = idxx; 
        end
    case 'soft_systematic_resampling'
        [idx, wk] = rs_soft_systematic(wk);
   otherwise
      error('Resampling strategy not implemented')
end;

xk = xk(:,idx);                    % extract new particles
switch resampling_strategy
       case 'soft_systematic_resampling'
            fprintf('soft_systematic_resampling');
   otherwise
      wk = repmat(1/Ns, 1, Ns); 
end;
        
% wk = repmat(1/Ns, 1, Ns);          % now all particles have the same weight

return;  % bye, bye!!!

function [ndx, neww] = rs_soft_systematic(w, bratio, rratio)
% function [ndx, neww] = rs_soft_systematic(w, bratio, rratio)
%
% INPUT
% w = normalised weights,
% bratio = softness of resampling -- 0<bratio<=1
% rratio = proportion of stochastic resammpling -- rratio>1
%
% bratio controls the number of particles spawned by heavy particles
%  if bratio is small, the particles tend more to stay heavy rather than
%  spawning
%
% rratio controls the number of particles that are considered candidates for
%  resampling when eliminating the light particles
%
% OUTPUT
% ndx = resampled particle indexes
% neww = new weights
%
% Victoria University of Wellington
% Paul Teal,  8 August 2012
% modified to properly treat tails, P Teal, Thursday 5 September 2013
% modified P Choppala, 3 June 2014

% P Choppala, P Teal, M Frean, IEEE SSP Workshop, GoldCoast, AUS, 2014
% Soft systematic resampling
% Victoria University of Wellington, New Zealand


if nargin <2
  bratio = 0.9;    %   0<bratio<=1
end

if nargin <3
  rratio = 2.5;  %     rratio>1
end

N = length(w); % no. of particles

% Soft resampling
[val,ind] = sort(w,'descend');
tmp = max(1, floor(N*bratio*val));

cc=1;
for p1=1:N
  ndx(cc:cc+tmp(p1)-1) = ind(p1);
  neww(cc:cc+tmp(p1)-1) = val(p1)/tmp(p1);
  cc=cc+tmp(p1);
end
M=length(ndx);

% Soft systematic ressampling
if M>N
  Noverflow = M - N;
  Nsmall    = min(M,round(rratio*Noverflow));
  Nresamp   = Nsmall - (M-N);
  light_ndxes = M-Nsmall+1:M;
  resamp_ndxes = M-Nsmall+1:N;

  ws=neww(light_ndxes);
  neww2=sum(ws)/Nresamp;
  ws=ws/sum(ws);
  ndx3=rs_systematic(ws, Nresamp);
  ndx(resamp_ndxes) = ndx(light_ndxes(ndx3));
  neww(resamp_ndxes) = neww2;
end

ndx=ndx(1:N);
neww=neww(1:N);
return;


function [indx,neww]=rs_systematic(w, D)
% function [indx,neww]=rs_systematic(w, D)
% Systematic resampler
% Allows the possibility to draw fewer samples than the no. of weights
% Doucet et al., SMC methods in practice, 2001.
% Arulampalam et al., Tutorial paper, 2002.
%
% Victoria University of Wellington,
% P Teal, Tuesday 4 September 2013
% modified P Choppala, Tue 3 June 2014

N=length(w);

if nargin<2
  D = N;
end

Q=cumsum(w);
indx=zeros(1,D);

u=([0:N-1]+rand(1))/N;
j=1;

for i=1:D
  while (Q(j)<u(i))
    j=j+1;
  end
  indx(i)=j;
end

neww=ones(1,N)/N;
return;

