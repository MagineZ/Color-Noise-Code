function [Y_os,eta] = optimal_shrinkage_color(Y,loss,r_p)
%Optimal singular value shrinkage over color noise
%=========input==================== 
% Y : Noisy data matrix;
% loss: = 'fro', 'op', 'nuc'
%=========output===================
% Y_os: denoised matrix
% eta: shrinked singular values
% Pei-Chun Su, 11/2021

[p,n] = size(Y);
transpose = 0;
if p>n
    Y = Y';
    transpose = 1;
end
[p,n] = size(Y);
[U,s,V] = svd(Y);
s = diag(s);
w = 10^(-1.9/3);
u = eig(Y'*Y); u = sort(u,'descend');
lab = eig(Y*Y'); lab = sort(lab,'descend');
xx = lab(1:r_p)./lab(2:(r_p+1))-1;

r_p = max(find(xx>=w));

%if r_p>6
%    r_p = 6;
%end

%r_p 

eta = zeros(1,length(lab));
    for j = 1:r_p
            
            m1 = (1/p *sum(1./(lab(r_p+1:end)-u(j))));
            dm1 = (1/p *sum(1./(lab(r_p+1:end)-u(j)).^2));
            m2 = (1/n *sum(1./(u(r_p+1:end)-u(j))));
            dm2 = (1/n *sum(1./(u(r_p+1:end)-u(j)).^2));
            Tau = u(j)*m1*m2; dTau = m1*m2 + u(j)*dm1*m2 + u(j)*m1*dm2;
            %d = sqrt(wt_sigma - hSigma(j));
            d = 1/sqrt(u(j)*m1*m2);
            %a1 = dtheta/(wt_sigma*theta);
            %a2 = d^2*dtheta/wt_sigma^3;
            a1 = m2/(d^2*dTau); a2 = m1/(d^2*dTau);
            if loss == "fro"
                eta(j) = d*sqrt(a1*a2);
            elseif loss == "op"
                eta(j) = d;
            elseif loss == "nuc"
                eta(j) = abs(d*(sqrt(a1*a2)- sqrt((1-a1)*(1-a2))));
            elseif loss == "rank"
                eta(j) = s(j);
            end
            
    end
Y_os = U*diag(eta)*V(:,1:p)';
if transpose
Y_os = Y_os';
end
end