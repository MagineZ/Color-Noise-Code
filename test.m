clear; close all;
cd('C:\Users\NTU_Math\Documents\prelim\color noise code')
%x = [4,2.1]; gamma = 1; name1 = 'Unif[1,10]';name2 = 'Mix2.2';
%x = [3,1.8];gamma = 0.5;name1 = 'Unif[1,10]';name2 = 'Mix2.2';
%x = [4,1.5];gamma = 1;name1 = 'Marcenko-Pastur';name2 = 'Marcenko-Pastur';
x = [3,1.3];gamma = 0.5;name1 = 'Marcenko-Pastur';name2 = 'Marcenko-Pastur';
r = length(x);
cm = 10; % student t 10 's moment
c = 2;
kk = 4; % k for imputation
smth = '';

%name1 = 'Marcenko-Pastur';
%name1 = 'Mix2';
%name1 = 'Unif[1,10]';
%name1 = 'PaddedIdentity';
%name1 = 'Chi10';
%name1 = 'Fisher3n';
%name1 = 'Triangle';



%name2 = 'Marcenko-Pastur';
%name2 = 'Mix2';
%name2 = 'Mix2.2';
%name2 = 'Unif[1,10]';
%name2 = 'PaddedIdentity';
%name2 = 'Chi10';
%name2 = 'Fisher3n';
%name2 = 'Triangle';
method = 'imp';
sh = 'op';

if sh == "fro"
    err_method = 'fro';
elseif sh == "op"
    err_method = 2;

end
ps = 200:200:1000;
T =50;

mF = [];mO = [];mN = [];
sF = [];sO = [];sN = [];
mF0 = [];mO0 = [];mN0 = [];
sF0 = [];sO0 = [];sN0 = [];
mS0 = [];mSw = [];mSi = [];
sS0 = [];sSw = [];sSi = [];
mwF = [];mwO = [];mwN = [];
swF = [];swO = [];swN = [];
mD1 = [];mA1 = [];mD2 = [];mA2 = [];
mrSN = [];mrp = [];mrold = [];mrW = [];
mDSN1 = [];mASN1 = [];mDSN2 = [];mASN2 = [];
mDW1 = [];mAW1 = [];mDW2 = [];mAW2 = [];
mD01 = [];mA01 = [];mD02 = [];mA02 = [];
for i = 1:length(ps)
    n = ps(i);
    errF = [];errO = [];errN = [];errD = [];
    errF0 = [];errO0 = [];errN0 = [];
    err_S0 = [];err_Sw = [];err_Si = [];
    err_wF = []; err_wO = []; err_wN = [];
    errA = []; err_rSN = [];err_rp = [];
    errDSN = []; errDW = []; errD0 = [];
    for t  = 1:T
        
        p = floor(n*gamma);
        G = randn(n,n);
        [V0, ~] =qr(G);
        V0 = V0(:,1:r);
        
        G = randn(p,p);
        [U0, ~] = qr(G);
        U0 = U0(:,1:r);
        
        X = U0 * diag(x) * V0';
        N = trnd(cm,p,n)./(sqrt(1.25)*sqrt(n));
        %student t 10/poisson mean 0/

        [Q1,~] = qr(randn(p,p));% modified QR decomposition
        [Q2,~] = qr(randn(n,n));
        [~,fT1] = Noise(p,p,name1,Q1);
        [~,fT2] = Noise(n,n,name2,Q2);
        A = Q1*fT1*Q1';
        B = Q2*fT2*Q2';
        sigma1 = sqrt(sum(diag(fT1).^2)/p);
        sigma2 = sqrt(sum(diag(fT2).^2)/n);
        A = A./sigma1; B = B./sigma2; 
        Y  = X + A*N*B;

        [U,hx,V] = svd(Y);
        hx = diag(hx);
        %[ZZ,~] = Noise(p,n^2,name);
        
     
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %[~, ~, r_p] = adaptiveHardThresholding(Y, k, 'i');
        %[~,eta_f] = optimal_shrinkage_color2(Y,sh,r_p,method);
        [~,eta_f,~,~] = optimal_shrinkage_color3(Y,sh,kk,method,cm-2,c);
        
        hX_f = U*diag(eta_f)*V(:,1:p)';
        err_f = norm(X-hX_f,err_method);
        errF = [errF; err_f];
        %[~,eta_o] = optimal_shrinkage_color2(Y,'op',r_p,method);
        [~,eta_o,r_p,eta_a] = optimal_shrinkage_color3(Y,'op',kk,method,cm,c);
        a = [];
        for k = 1:length(x)
            a = [a, (U(:,k)'*U0(:,k)* V(:,k)'*V0(:,k))];
        end
        errD = [errD;abs(eta_o(1:length(x))-x)];
        errA = [errA;abs(eta_a(1:length(x))-a)];
        eta_f0 = optimal_shrinkage(hx,gamma,sh);
        hX_f0 = U*diag(eta_f0)*V(:,1:p)';
        err_f0 = norm(X-hX_f0,err_method);
        errF0 = [errF0; err_f0];
        errD0 = [errD0;abs(eta_f0(1:length(x))'-x)];
      

        [hX_si, ~, rSN] = adaptiveHardThresholding(Y, kk, 'i');
        %hy = [hx(1),hx(2),zeros(1,length(hx)-2)];
        %hX_si = U*diag(hy)*V(:,1:p)';
        err_rSN = [err_rSN;rSN];err_rp = [err_rp;r_p];
        err_si = norm(X-hX_si,err_method);
        err_Si = [err_Si; err_si];
        eta_sn = svd(hX_si);
        errDSN = [errDSN;abs(eta_sn(1:length(x))'-x)];
        %Whitening
        invSigma2 = Q1*(pinv(fT1))*Q1';
        Sigma2 = A;
        wY = invSigma2*Y;
        [wU,wD,wV] = svd(wY);
        hD = optimal_shrinkage(diag(wD),gamma,sh,1);
        whX_fro = wU*diag(hD)*wV(:,1:p)'; hX_fro = Sigma2*whX_fro; 
        err_fro = norm(X-hX_fro,err_method);
        
        err_wF = [err_wF; err_fro];
        errDW = [errDW;abs(hD(1:length(x))'-x)];
 
        
    end
    mF = [mF, errF];sF = [sF, std(errF)];
    mF0 = [mF0, errF0];sF0 = [sF0, std(errF0)];
    mSi = [mSi,err_Si];sSi = [sSi,std(err_Si)];
    mwF = [mwF,err_wF];swF = [swF, std(err_wF)];
    mrSN = [mrSN, err_rSN];mrp = [mrp, err_rp];
    
    mD1 = [mD1, errD(:,1)];mD2 = [mD2, errD(:,2)];
    mDSN1 = [mDSN1, errDSN(:,1)];mDSN2 = [mDSN2, errDSN(:,2)];
    mDW1 = [mDW1, errDW(:,1)];mDW2 = [mDW2, errDW(:,2)];
    mD01 = [mD01, errD0(:,1)];mD02 = [mD02, errD0(:,2)];
    mA1 = [mA1, errA(:,1)];mA2 = [mA2, errA(:,2)];
    
end
%boxplot(ps, mF)
errorbar(ps,median(mF),median(mF)-quantile(mF,.25),quantile(mF,.75)-median(mF),'k','Linewidth', 5);
hold on
errorbar(ps,median(mSi),median(mSi)-quantile(mSi,.25),quantile(mSi,.75)-median(mSi),'g','Linewidth', 4);
errorbar(ps,median(mF0),median(mF0)-quantile(mF0,.25),quantile(mF0,.75)-median(mF0),'m','Linewidth', 3);
errorbar(ps,median(mwF),median(mwF)-quantile(mwF,.25),quantile(mwF,.75)-median(mwF),'c','Linewidth', 2);
%errorbar(ps, median(mF),median(sF),'k','Linewidth', 5);
%errorbar(ps,median(mF0),median(sF0), 'm','Linewidth', 5);
%errorbar(ps,median(mwF),median(swF),'#D95319','Linewidth', 5);
%legend('Our algorithm','Original Optimal Shrinkage','ScreeNot', 'Whitening','Location','northwest')
set(gca,'FontSize',20)
xlabel('n');
ylabel('Error')
axis tight
ylim([1.1,3.5])
%ylim([1.02,1.16])
%ylim([0.96,1.04])
%set(gcf, 'Position', [0,0,1300,800]);
saveas(gcf,['norm_error_',smth,sh,'_',method],'png');
hold off;

errorbar(ps,median(abs(mD1./x(1)*100)),median(abs(mD1./x(1)*100))-quantile(abs(mD1./x(1)*100),.25),quantile(abs(mD1./x(1)*100),.75)-median(abs(mD1./x(1)*100)),'k','Linewidth', 5);
hold on;
errorbar(ps,median(abs(mD2./x(2)*100)),median(abs(mD2./x(2)*100))-quantile(abs(mD2./x(2)*100),.25),quantile(abs(mD2./x(2)*100),.75)-median(abs(mD2./x(2)*100)),'--k','Linewidth', 5);
errorbar(ps,median(abs(mDSN1./x(1)*100)),median(abs(mDSN1./x(1)*100))-quantile(abs(mDSN1./x(1)*100),.25),quantile(abs(mDSN1./x(1)*100),.75)-median(abs(mDSN1./x(1)*100)),'g','Linewidth', 4);
errorbar(ps,median(abs(mDSN2./x(2)*100)),median(abs(mDSN2./x(2)*100))-quantile(abs(mDSN2./x(2)*100),.25),quantile(abs(mDSN2./x(2)*100),.75)-median(abs(mDSN2./x(2)*100)),'--g','Linewidth', 4);
errorbar(ps,median(abs(mD01./x(1)*100)),median(abs(mD01./x(1)*100))-quantile(abs(mD01./x(1)*100),.25),quantile(abs(mD01./x(1)*100),.75)-median(abs(mD01./x(1)*100)),'m','Linewidth', 3);
errorbar(ps,median(abs(mD02./x(2)*100)),median(abs(mD02./x(2)*100))-quantile(abs(mD02./x(2)*100),.25),quantile(abs(mD02./x(2)*100),.75)-median(abs(mD02./x(2)*100)),'--m','Linewidth', 3);
errorbar(ps,median(abs(mDW1./x(1)*100)),median(abs(mDW1./x(1)*100))-quantile(abs(mDW1./x(1)*100),.25),quantile(abs(mDW1./x(1)*100),.75)-median(abs(mDW1./x(1)*100)),'c','Linewidth', 2);
errorbar(ps,median(abs(mDW2./x(2)*100)),median(abs(mDW2./x(2)*100))-quantile(abs(mDW2./x(2)*100),.25),quantile(abs(mDW2./x(2)*100),.75)-median(abs(mDW2./x(2)*100)),'--c','Linewidth', 2);

%plot(ps,mD./x*100,'Linewidth', 3);
%legend('error 1','error 2','error 3')
set(gca,'FontSize',20)
xlabel('n');
ylabel('Error rate (%)')
axis tight
ylim([0,25])
saveas(gcf,['d_error_',smth,method],'png');
hold off;

errorbar(ps,median(abs(mA1)),median(abs(mA1))-quantile(abs(mA1),.25),quantile(abs(mA1),.75)-median(abs(mA1)),'b','Linewidth', 5);
hold on;
errorbar(ps,median(abs(mA2)),median(abs(mA2))-quantile(abs(mA2),.25),quantile(abs(mA2),.75)-median(abs(mA2)),'r','Linewidth', 5);
legend('error 1','error 2','error 3')
set(gca,'FontSize',20)
xlabel('n');
ylabel('Angle Error')
axis tight
ylim([0,0.5])
saveas(gcf,['a_error_',smth,method],'png');
hold off;



plot(ps, sum(mrSN==2)./50*100,'g','Linewidth', 5);
hold on;
plot(ps, sum(mrSN==1)./50*100,':g','Linewidth', 5);
plot(ps,sum(mrp==2)./50*100,'k','Linewidth', 5);
plot(ps,sum(mrp==1)./50*100,':k','Linewidth', 5);
set(gca,'FontSize',20)
xlabel('n');
ylabel('%')
axis tight
%set(gcf, 'Position', [0,0,1300,800]);
saveas(gcf,['rank_',smth ,method],'png');
hold off;
%{
plot(ps,mS(:,1),'Linewidth',7)
hold on
plot(ps,mS(:,2),'Linewidth',7)
plot(ps,mS(:,3),'Linewidth',7)
plot(ps,mS(:,4),'Linewidth',7)
plot(ps,mS(:,5),'Linewidth',7)
legend('error 1', 'error 2','error 3','error 4','error 5')
set(gca,'FontSize',20)
xlabel('n')
ylabel('Error')

mA1 = abs(mA1);
plot(ps,mA1(:,1),'Linewidth',2)
hold on
plot(ps,mA1(:,2),'Linewidth',2)
plot(ps,mA1(:,3),'Linewidth',2)
plot(ps,mA1(:,4),'Linewidth',2)
plot(ps,mA1(:,5),'Linewidth',2)
legend('error 1', 'error 2','error 3','error 4','error 5')
set(gca,'FontSize',20)
xlabel('n')
ylabel('Error')
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
xlabel('n')
ylabel('Error')
axis tight
title('Right Cosine Error')

%}