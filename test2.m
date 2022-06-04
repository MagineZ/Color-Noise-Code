clear; close all;
f1 = figure;f2 = figure; f3 = figure; f4 = figure;
x_0 = 10:-1:5;
r = length(x_0);
k = 4*r;
T = 20;
ps = 100:100:600;
gamma = 1;
mF = [];mO = [];mN = [];mS = [];
sF = [];sO = [];sN = [];sS = [];
mS0 = [];mSw = [];mSi_f = [];mSi_o = [];mSi_n = [];
sS0 = [];sSw = [];sSi_f = [];sSi_o = [];sSi_n = [];
mwF = [];mwO = [];mwN = [];
mF0 = [];mO0 = [];mN0 = [];
swF = [];swO = [];swN = [];
mA1 = [];mA2 = [];mD = [];
sF0 = [];sO0 = [];sN0 = [];
mshF = []; mshO = []; mshN = [];
sshF = []; sshO = []; sshN = [];
%rng(100)
%name = 'Marcenko-Pastur';
%name = 'Mix2';
%name = 'Unif[1,10]';
%name = 'PaddedIdentity';
name = 'Chi10';
%name = 'Fisher3n';
%name = 'Triangle';

for i = 1:length(ps)
    p = ps(i);
    x = [x_0, zeros(1,p - length(x_0))];
    errF = [];errO = [];errN = [];errSigma = [];
    err_S0 = [];err_Sw = [];err_Si_f = [];err_Si_o = [];err_Si_n = [];
    err_wF = []; err_wO = []; err_wN = [];
    err_F0 = []; err_O0 = []; err_N0 = [];
    err_wF2 = [];
    errA1 = []; errA2 = [];errD = [];
    err_sh_F = [];err_sh_O = [];err_sh_N = [];
    for t  = 1:T
        
        n = ceil(p/gamma);
        w = n^(-1/3);
        G = randn(n,n);
        [A, ~, ~] = svd(G);
        A = A(:,1:p);
        G = randn(p,p);
        [B, ~, ~] = svd(G);
        %B = B(:,1:r);
        X = B * diag(x) * A';
        %Sigma = ones(1,p);%sort(abs(randn(1,p)),'descend');
        [Q,~] = svd(randn(p,n));
        %Q = diag(ones(1,p));
        [Z,fT] = Noise(p,n,name,Q);
        Sigma = Q*fT.^2*Q';
        
        [~,sigma] = svd(Sigma);
        sigma = diag(sigma);
        sigma = sort(sigma,'descend');
        Y = X + Z;
        [U,hx,V] = svd(Y);
   
        %[ZZ,~] = Noise(p,n^2,name);
        
     
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        eta_f0 = zeros(1,p);
        eta_o0 = zeros(1,p);
        eta_n0 = zeros(1,p);
        eta_f = zeros(1,p);
        eta_o = zeros(1,p);
        eta_n = zeros(1,p);
        u = eig(Y'*Y); u = sort(u,'descend');
        lambda = eig(Y*Y'); lambda = sort(lambda,'descend');
        
        xx = lambda(1:50)./lambda(2:51)-1;
        %r_p = max(find(xx>=w));
        r_p = r;
        a1_col = []; a2_col = []; d_col = [];
        f_col = []; o_col = []; n_col = [];

        for j = 1:r_p
            dtheta = -(1/p *sum(1./(u(r_p+1:end)-u(j)).^2))^(-1);
            m1 = (1/p *sum(1./(lambda(r_p+1:end)-u(j))));
            dm1 = (1/p *sum(1./(lambda(r_p+1:end)-u(j)).^2));
            m2 = (1/n *sum(1./(u(r_p+1:end)-u(j))));
            dm2 = (1/n *sum(1./(u(r_p+1:end)-u(j)).^2));
            Tau = u(j)*m1*m2; dTau = m1*m2 + u(j)*dm1*m2 + u(j)*m1*dm2;
         
            d = 1/sqrt(Tau);
            a1 = m2/(d^2*dTau); a2 = m1/(d^2*dTau);
            c1 = (U(:,j)'*B(:,j)); c2 = (V(:,j)'*A(:,j)); 
            
            eta_f(j) = d*sqrt(a1*a2);
            eta_o(j) = d;
            eta_n(j) = abs(d*(sqrt(a1*a2)- sqrt((1-a1)*(1-a2))));
            
            eta_f0(j) = x(j)*(c1*c2);
            eta_o0(j) = x(j);
            eta_n0(j) = abs(x(j)*((c1*c2)- sqrt((1-c1^2)*(1-c2^2))));

            err_a1 = abs(abs(U(:,j)'*B(:,j)) - sqrt(abs(a1)));
            err_a2 = abs(abs((V(:,j)'*A(:,j))) - sqrt(abs(a2)));
            
            a1_col = [a1_col,err_a1];
            a2_col = [a2_col,err_a2];
            d_col = [d_col,d-x(j)];
            f_col = [f_col, abs(eta_f(j)-eta_f0(j))];
            o_col = [o_col, abs(eta_o(j)-eta_o0(j))];
            n_col = [n_col, abs(eta_n(j)-eta_n0(j))];
        end
        
        
        hX_f = U*diag(eta_f)*V(:,1:p)';
        err_f = norm(X-hX_f,'fro');
        hX_o = U*diag(eta_o)*V(:,1:p)';
        err_o = max(svd(X-hX_o));
        hX_n = U*diag(eta_n)*V(:,1:p)';
        err_n = sum(svd(X-hX_n));
        errF = [errF; err_f];
        errO = [errO; err_o];
        errN = [errN; err_n];
        errA1 = [errA1;a1_col(1:r)];
        errA2 = [errA2;a2_col(1:r)];
        errD = [errD;d_col(1:r)];
        err_sh_F = [err_sh_F; f_col(1:r)];
        err_sh_O = [err_sh_O; o_col(1:r)];
        err_sh_N = [err_sh_N; n_col(1:r)];
        %ScreeNot
        %[hX_s0, ~, ~] = adaptiveHardThresholding(Y, k, '0');
        %err_s0 = norm(X-hX_s0,'fro');
        %[hX_sw, ~, ~] = adaptiveHardThresholding(Y, k, 'w');
        %err_sw = norm(X-hX_sw,'fro');
        [hX_si, ~, ~] = adaptiveHardThresholding(Y, k, 'i');
        err_si_f = norm(X-hX_si,'fro');
        err_si_o  = max(svd(X-hX_si));
        err_si_n  = sum(svd(X-hX_si));
        %err_S0 = [err_S0; err_s0];
        %err_Sw = [err_Sw; err_sw];
        err_Si_f = [err_Si_f; err_si_f];
        err_Si_o = [err_Si_o; err_si_o];
        err_Si_n = [err_Si_n; err_si_n];
        %Whitening
        [sU,sD] = svd(Sigma);
        %[sU,sD] = svd(sigmahat);
        invSigma2 = sU*(pinv(sD).^(1/2))*sU';
        Sigma2 = sU*(sD.^(1/2))*sU';
        wY = invSigma2*Y;
        [wU,wD,wV] = svd(wY);
        hD = optimal_shrinkage(diag(wD),gamma,'fro');
        whX_fro = wU*diag(hD)*wV(:,1:p)'; hX_fro = Sigma2*whX_fro; 
        err_fro = norm(X-hX_fro,'fro');
        
        
        
        hD = optimal_shrinkage(diag(wD),gamma,'op');
        whX_op = wU*diag(hD)*wV(:,1:p)'; hX_op = Sigma2*whX_op; 
        err_op = max(svd(X-hX_op));
        
        hD = optimal_shrinkage(diag(wD),gamma,'nuc');
        whX_nuc = wU*diag(hD)*wV(:,1:p)'; hX_nuc = Sigma2*whX_nuc; 
        err_nuc = sum(svd(X-hX_nuc));
        
        
        err_wF = [err_wF; err_fro];
        err_wO = [err_wO; err_op];
        err_wN = [err_wN; err_nuc];
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [U0,D0,V0] = svd(Y);
        hD0 = optimal_shrinkage(diag(D0),gamma,'fro');
        X0_fro = U0*diag(hD0)*V0(:,1:p)'; 
        err0_fro = norm(X-X0_fro,'fro');
        
        
        
        hD = optimal_shrinkage(diag(D0),gamma,'op');
        X0_op = U0*diag(hD0)*V0(:,1:p)'; 
        err0_op = max(svd(X-X0_op));
        
        hD = optimal_shrinkage(diag(D0),gamma,'nuc');
        X0_nuc = U0*diag(hD0)*V0(:,1:p)'; 
        err0_nuc = sum(svd(X-X0_nuc));
        
        
        err_F0 = [err_F0; err0_fro];
        err_O0 = [err_O0; err0_op];
        err_N0 = [err_N0; err0_nuc];



       
    end
    mF = [mF, errF];mO = [mO, errO];mN = [mN, errN];
    sF = [sF, std(errF)];sO = [sO, std(errO)];sN = [sN, std(errN)];
    %mS0 = [mS0, err_S0];mSw = [mSw, err_Sw];
    mSi_f = [mSi_f,err_Si_f];mSi_o = [mSi_o,err_Si_o];mSi_n = [mSi_n,err_Si_n];
    %sS0 = [sS0,std(err_S0)];sSw = [sSw,std(err_Sw)];
    sSi_f = [sSi_f,std(err_Si_f)];sSi_o = [sSi_o,std(err_Si_o)];sSi_n = [sSi_n,std(err_Si_n)];
    mwF = [mwF,err_wF];mwO = [mwO, err_wO];mwN = [mwN,err_wN];
    swF = [swF, std(err_wF)];swO = [swO, std(err_wO)];swN = [swN, std(err_wN)];
    mF0 = [mF0,err_F0];mO0 = [mO0, err_O0];mN0 = [mN0,err_N0];
    sF0 = [sF0, std(err_F0)];sO0 = [sO0, std(err_O0)];swN = [sN0, std(err_N0)];
    mD = [mD;median(errD)];mA1 = [mA1; median(errA1)];mA2 = [mA2; median(errA2)];
    mshF = [mshF;median(err_sh_F)]; mshO = [mshO;median(err_sh_O)]; mshN = [mshN;median(err_sh_N)];
    
end
%boxplot(ps, mF)

%semilogy(ps, median(mO),'Linewidth', 5);
%plot(ps, log(median(mO)),'Linewidth', 5);
%{
plot(ps,abs(mS(:,1)));
hold on;
%plot(ps,median(mSi),'Linewidth', 5);
%plot(ps,median(mwF),'Linewidth', 5);
%legend('Our algorithm','ScreeNot', 'Whitening')
set(gca,'FontSize',20)
xlambdael('n');
ylambdael('Error')
%}
figure(f1);
plot(ps, median(mF),'o-','Markersize',40,'Linewidth',5);
hold on;
plot(ps,median(mSi_f),'^-','Markersize',40,'Linewidth', 5);
plot(ps,median(mwF),'s-','Markersize',40,'Linewidth', 5);
plot(ps,median(mF0),'+-','Markersize',40,'Linewidth', 5);
legend('Our algorithm','ScreeNot:Imputation', 'Oracle Whitening', 'Optimal Shrinkage: Matan','Interpreter','latex')
set(gca,'FontSize',30,'TickLabelInterpreter','latex')
xlabel('n','Interpreter','latex');
ylabel('Error','Interpreter','latex')
hold off;

figure(f2);
plot(ps,mshF,'Linewidth',7)
hold on
legend('error 1', 'error 2','error 3','error 4','error 5','error 6','Interpreter','latex')
set(gca,'FontSize',30,'TickLabelInterpreter','latex')
xlabel('n','Interpreter','latex')
ylabel('Error','Interpreter','latex')

figure(f3);
plot(ps,mshO,'Linewidth',7)
hold on
legend('error 1', 'error 2','error 3','error 4','error 5','error 6','Interpreter','latex')
set(gca,'FontSize',30,'TickLabelInterpreter','latex')
xlabel('n','Interpreter','latex')
ylabel('Error','Interpreter','latex')

figure(f4);
plot(ps,mshN,'Linewidth',7)
hold on
legend('error 1', 'error 2','error 3','error 4','error 5','error 6','Interpreter','latex')
set(gca,'FontSize',30,'TickLabelInterpreter','latex')
xlabel('n','Interpreter','latex')
ylabel('Error','Interpreter','latex')
%{
plot(ps,mS(:,1),'Linewidth',7)
hold on
plot(ps,mS(:,2),'Linewidth',7)
plot(ps,mS(:,3),'Linewidth',7)
plot(ps,mS(:,4),'Linewidth',7)
plot(ps,mS(:,5),'Linewidth',7)
legend('error 1', 'error 2','error 3','error 4','error 5')
set(gca,'FontSize',20)
xlambdael('n')
ylambdael('Error')

mA1 = abs(mA1);
plot(ps,mA1(:,1),'Linewidth',2)
hold on
plot(ps,mA1(:,2),'Linewidth',2)
plot(ps,mA1(:,3),'Linewidth',2)
plot(ps,mA1(:,4),'Linewidth',2)
plot(ps,mA1(:,5),'Linewidth',2)
legend('error 1', 'error 2','error 3','error 4','error 5')
set(gca,'FontSize',20)
xlambdael('n')
ylambdael('Error')
axis tight
title('Left Cosine Error')
%}
%{
mA2 = abs(mA2);
plot(ps,mA2(:,1),'Linewidth',2)
hold on
plot(ps,mA2(:,2),'Linewidth',2)
plot(ps,mA2(:,3),'Linewidth',2)
plot(ps,mA2(:,4),'Linewidth',2)
plot(ps,mA2(:,5),'Linewidth',2)
legend('error 1', 'error 2','error 3','error 4','error 5')
set(gca,'FontSize',20)
xlambdael('n')
ylambdael('Error')
axis tight
title('Right Cosine Error')

%}