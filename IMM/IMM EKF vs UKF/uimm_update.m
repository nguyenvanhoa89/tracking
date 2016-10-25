%IMM_UPDATE  Interacting Multiple Model (IMM) Filter update step
%
% Syntax:
%   [X_i,P_i,w,X,P] = IMM_UPDATE(X_p,P_p,c_j,ind,dims,Y,H,R, X_hat, X_dev,dt)
%
% In:
%   X_p  - Cell array containing N^j x 1 mean state estimate vector for
%          each model j after prediction step
%   P_p  - Cell array containing N^j x N^j state covariance matrix for 
%          each model j after prediction step
%   c_j  - Normalizing factors for mixing probabilities
%   ind  - Indices of state components for each model as a cell array
%   dims - Total number of different state components in the combined system
%   Y    - Dx1 measurement vector.
%   H    - Measurement matrices for each model as a cell array.
%   R    - Measurement noise covariances for each model as a cell array.
%   X_hat - transformed sampling points
%   X_dev - transformed deviations
%   dt - 1 cycle 
%
% Out:
%   X_i  - Updated state mean estimate for each model as a cell array
%   P_i  - Updated state covariance estimate for each model as a cell array
%   w    - Estimated weights of each model
%   X    - Combined state mean estimate
%   P    - Combined state covariance estimate
%   
% Description:
%   IMM filter measurement update step.
%
% See also:
%   IMM_PREDICT, IMM_SMOOTH, IMM_FILTER

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

function [X_i,P_i,w,X,P] = uimm_update(X_p,P_p,c_j,ind,dims,Y,H,R, X_hat, X_dev,dt)
    % Number of models 
    m = length(X_p);

    % Space for update state mean, covariance and likelihood of measurements
    X_i = cell(1,m);
    P_i = cell(1,m);
    lambda = zeros(1,m);

    % Update for each model
    for i = 1:m
        % Update the state estimates
        if isa(H{i},'function_handle')
            [X_i{i}, P_i{i}, ~, ~, lambda(i)] = ukf_update(X_p{i},P_p{i},Y,H{i},R{i}, X_hat{i}, X_dev{i},dt);
        else
            [X_i{i}, P_i{i}, ~, ~, lambda(i)] = kf_update(X_p{i},P_p{i},Y,H{i},R{i});
        end
        
    end
    
    % Calculate the model probabilities
    w = zeros(1,m); 
    c = sum(lambda.*c_j);
    w = c_j.*lambda/c;
    
    % Output the combined updated state mean and covariance, if wanted.
    if nargout > 3
        % Space for estimates
        X = zeros(dims,1);
        P = zeros(dims,dims);
        % Updated state mean
        for i = 1:m
            X(ind{i}) = X(ind{i}) + w(i)*X_i{i};
        end
        % Updated state covariance
        for i = 1:m
            P(ind{i},ind{i}) = P(ind{i},ind{i}) + w(i)*(P_i{i} + (X_i{i}-X(ind{i}))*(X_i{i}-X(ind{i}))');
        end
    end
end
    