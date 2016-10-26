%IMM_PREDICT  Interacting Multiple Model (IMM) Filter prediction step
%
% Syntax:
%   [X_p,P_p,c_j,X_hat,X_dev] = IMM_PREDICT(X_ip,P_ip,w,p_ij,ind,dims,F,Q,dt)
%
% In:
%   X_ip  - Cell array containing N^j x 1 mean state estimate vector for
%           each model j after update step of previous time step
%   P_ip  - Cell array containing N^j x N^j state covariance matrix for 
%           each model j after update step of previous time step
%   w -     weight Vector containing the model probabilities at previous time step
%   p_ij  - Model transition probability matrix
%   ind   - Indexes of state components for each model as a cell array
%   dims  - Total number of different state components in the combined system
%   F     - State transition matrices for each model as a cell array.
%   Q     - Process noise matrices for each model as a cell array.
%   dt - 1 cycle 
%
% Out:
%   X_p   - Predicted state mean for each model as a cell array
%   P_p   - Predicted state covariance for each model as a cell array
%   c_j   - Normalizing factors for mixing probabilities
%   X_hat - transformed sampling points
%   X_dev - transformed deviations
% %   X     - Combined predicted state mean estimate
% %   P     - Combined predicted state covariance estimate
%   
% Description:
%   IMM filter prediction step.
%
% See also:
%   IMM_UPDATE, IMM_SMOOTH, IMM_FILTER

% History:
%   25.10.2016 Hoa updated per his version.
%   01.11.2007 JH The first official version.
%
% Copyright (C) 2007 Jouni Hartikainen
%
% $Id: imm_update.m 111 2007-11-01 12:09:23Z jmjharti $
%
% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.

function [X_p,P_p,c_j,X_hat,X_dev] = uimm_predict(X_ip,P_ip,w,p_ij,ind,dims,F,Q,dt)
    % Number of models 
    m = length(X_ip);
    
    % Default values for state mean and covariance
    MM_def = zeros(dims,1);
    PP_def = diag(20*ones(dims,1));

    % Normalizing factors for mixing probabilities
    c_j = zeros(1,m);
    for j = 1:m
        for i = 1:m
            c_j(j) = c_j(j) + p_ij(i,j).*w(i);
        end
    end

    % Mixing probabilities
    MU_ij = zeros(m,m);
    for i = 1:m
        for j = 1:m
            MU_ij(i,j) = p_ij(i,j) * w(i) / c_j(j);
        end
    end

    % Calculate the mixed state mean for each filter
    X_0j = cell(1,m);
    for j = 1:m
        X_0j{j} = zeros(dims,1);
        for i = 1:m
            X_0j{j}(ind{i}) = X_0j{j}(ind{i}) + X_ip{i}*MU_ij(i,j);
        end
    end
    
    % Calculate the mixed state covariance for each filter
    P_0j = cell(1,m);
    for j = 1:m
        P_0j{j} = zeros(dims,dims);
        for i = 1:m
            P_0j{j}(ind{i},ind{i}) = P_0j{j}(ind{i},ind{i}) + MU_ij(i,j)*(P_ip{i} + (X_ip{i}-X_0j{j}(ind{i}))*(X_ip{i}-X_0j{j}(ind{i}))');
        end
    end

    % Space for predictions
    X_p = cell(1,m);
    X_hat = cell(1,m);
    P_p = cell(1,m);
    X_dev = cell(1,m);

    % Make predictions for each model
    for i = 1:m
        [X_p{i}, X_hat{i}, P_p{i},X_dev{i}] = ukf_predict(X_0j{i}(ind{i}),P_0j{i}(ind{i},ind{i}),F{i},Q{i},dt);
    end

%     % Output the combined predicted state mean and covariance, if wanted.
%     if nargout > 3
%         % Space for estimates
%         X = zeros(dims,1);
%         P = zeros(dims,dims);
%         
%         % Predicted state mean
%         for i = 1:m
%             X(ind{i}) = X(ind{i}) + w(i)*X_p{i};
%         end
% 
%         % Predicted state covariance
%         for i = 1:m
%             P(ind{i},ind{i}) = P(ind{i},ind{i}) + w(i)*(P_p{i} + (X_ip{i}-X(ind{i}))*(X_i{i}-X(ind{i}))');
%         end
%     end
end
    