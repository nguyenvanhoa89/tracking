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
        
    case 'RPF'
        with_replacement = true;
        idx = randsample(1:Ns, Ns, with_replacement, wk);
        
        
    case 'soft_systematic_resampling'
        [idx, wk] = rs_soft_systematic(wk);
   otherwise
      error('Resampling strategy not implemented')
end;

                % extract new particles
switch resampling_strategy
       case 'soft_systematic_resampling'
            fprintf('soft_systematic_resampling');
             xk = xk(:,idx);    
       case 'RPF'
            fprintf('RPF');       
            S=xk * diag(wk) * xk'; %empirical covariance matrix 
            L=chol(S,'lower'); %the square root matrix of S 
            m =size(xk,1);
            epsilon=zeros(m,Ns); 
            A=(4/(m+2))^(1/(m+4)); 
            h=A*(Ns^(-1/(m+4))); 
            xk = xk(:,idx);     
            wk = repmat(1/Ns, 1, Ns); 
            for i=1:Ns 
                epsilon(:,i)=(h*L)*randn(1,m)'; 
                xk(:,i)=xk(:,i)+h.*(L*(epsilon(:,i))); 
            end
            
    otherwise
      xk = xk(:,idx);     
      wk = repmat(1/Ns, 1, Ns); 
end;
return;

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
