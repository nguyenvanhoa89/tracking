function [rk] = regime_transion(rkm1, p, Ns)
    % input
    % r: regime variable
    % p: transitional probability matrix
%     p = [0.98 0.02;0.02 0.98];
%     Ns = 100;
%     rkm1 = ones(Ns,1);
    rk = ones(Ns,1);
    S = size(p,1);  % S = number of models
    c = zeros(S,S);
    
    for i=1:S   
        for j=1:S
            if j == 1
                c (i,j) = p(i,j);
            else
                c (i,j) = c (i,j-1) + p(i,j);
            end
        end   
    end
    for n=1:Ns
        u = rand;
        i = rkm1(n);
        m = 1;
        while c(i,m) < u
            m = m + 1;
        end
        rk(n) = m;
    end
end
    