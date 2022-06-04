function [Y_os,eta,r_p,eta_a] = optimal_shrinkage_color3(Y,loss,k,method,cm,c)
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
w = c*max(p^(-2/3),p^(4/cm-1));
u = eig(Y'*Y); u = sort(u,'descend');
lab = eig(Y*Y'); lab = sort(lab,'descend');

fZ = createPseudoNoise(s, k, 'i');
%gamma = min( p/n, n/p);
%Topt = computeOptThreshold(fZ, gamma);
%r_p = sum(s>Topt)
xx = lab(1:k)./lab(2:(k+1))-1;
r_p = max(find(xx>w))
%r_p = max(find(lab(1:k)-fZ(1).^2>0));
%r_p = max(find(lab>5*(p^(-2/3)+p^(-0.6)+fZ(1)^2)));
ov = lab(1:r_p);
lab(1:p) = fZ.^2; u(1:p) = fZ.^2;

eta = zeros(1,length(lab));
eta_a = zeros(1,length(lab));
for j = 1:r_p

    if method == "cut"
        m1 = (1/p *sum(1./(lab((k+1):end)-ov(j))));
        dm1 = (1/p *sum(1./(lab((k+1):end)-ov(j)).^2));
        m2 = (1/n *sum(1./(u((k+1):end)-ov(j))));
        dm2 = (1/n *sum(1./(u((k+1):end)-ov(j)).^2));
    else
        m1 = (1/p *sum(1./(lab(1:end)-ov(j))));
        dm1 = (1/p *sum(1./(lab(1:end)-ov(j)).^2));
        m2 = (1/n *sum(1./(u(1:end)-ov(j))));
        dm2 = (1/n *sum(1./(u(1:end)-ov(j)).^2));
    end

    Tau = ov(j)*m1*m2; dTau = m1*m2 + ov(j)*dm1*m2 + ov(j)*m1*dm2;
    %d = sqrt(wt_sigma - hSigma(j));
    d = 1/sqrt(ov(j)*m1*m2);
    %a1 = dtheta/(wt_sigma*theta);
    %a2 = d^2*dtheta/wt_sigma^3;
    a1 = abs(m1/(d^2*dTau)); a2 = (m2/(d^2*dTau));

    if loss == "fro"
        eta(j) = d*sqrt(a1*a2);
    elseif loss == "op"
        eta(j) = d;
    elseif loss == "nuc"
        eta(j) = abs(d*(sqrt(a1*a2)- sqrt((1-a1)*(1-a2))));
    elseif loss == "rank"
        eta(j) = s(j);
    end
    eta_a(j) = sqrt(a1*a2);
    %if sqrt(a1*a2)<w
    %    eta(j) = 0;
    %    eta_a(j) =0;
    %end

end
r_p = sum(eta>0);
Y_os = U*diag(eta)*V(:,1:p)';
if transpose
    Y_os = Y_os';
end
end