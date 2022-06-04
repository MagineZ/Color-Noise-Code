%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [gamma_output,P_output]=eig_mestre(Rest,N,Kvec,err)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input parameters:
% Rest: Sample covariance matrix. Must be square and Hermitian
% N: Number of observations used to construct the sample covariance matrix
% Kvec: Vector containing the multiplicity of each true eigenvalue. If left
%       blank, an all ones vector is assumed.
% err:  Relative error tolerated when obtaining the mu parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output parameters:
% gamma_output: Column vector containing the estimated eigenvalues
% P_output: Tensor of dimension MxMxMbar (M is the observation dimension
%           and Mbar the number of true eigenvalues), so that, for each m,
%           P_output(:,:,m) contains the estimated projector to the m-th
%           eigen-subspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ® Xavier Mestre, 2008, all rights reserved
% For implementation details, please refer to: Xavier Mestre, "Improved
% estimation of eigenvalues of covariance matrices and their associated 
% subspaces using their sample estimates", IEEE Transactions on Information 
% Theory, vol. 54, no. 11, Nov. 2008.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gamma_output,P_output]=eig_mestre(Rest,N,Kvec,err)
M=size(Rest,1);       % Observation dimension
if nargin<3,
    Kvec=ones(1,M); % We assume that that all the true eigenvalues have multiplicity 1
    err=1e-6;       % Relative error for the calculation of the mu variables
end;
if nargin<4,
    err=1e-6;       % Relative error for the calculation of the mu variables
end;    
Mbar=length(Kvec);  % Number of different eigenvalues
gamma_output=zeros(Mbar,1);
P_output=zeros(M,M,Mbar);
if size(Rest,1)~=size(Rest,2),
    fprintf('Error: The sample covariance matrix must be square.\n')
    return
end;
if min(abs(Rest-Rest.'))>0,
    fprintf('Error: The sample covariance matrix must be Hermitian.\n')
    return
end;
if sum(Kvec)~=M,
    fprintf('Error: The sum of the entries of the multiplicity vector must be equal to the matrix dimension.\n')
    return
end;
if min(Kvec)<=0,
    fprintf('Error: All the entries of the multiplicity vector must be positive.\n')
    return
end;

c=M/N;

[E_trad,lambda_trad]=eig(Rest); % We construct the sample eigenvalues and eigenvectors
lambda_trad=real(diag(lambda_trad));    % Sample eigenvalues
[lambda_trad,index]=sort(lambda_trad);
E_trad=E_trad(:,index);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the mu variables, by the Newton-Rapson method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
muest=zeros(size(lambda_trad));
if c<1,
    if min(abs(diff(lambda_trad)))==0
        fprintf('Error: Sample covariance matrix contains non-zero eigenvalues with multiplicity higher than one.\n')
        return
    end;
    for m=1:M,
        if m>1,
            mutemp=(lambda_trad(m)+lambda_trad(m-1))/2;
            mutemp2=(2*lambda_trad(m)+lambda_trad(m-1))/3;
            while abs((mutemp-mutemp2)./mutemp)>err,
                mutemp2=mutemp;
                ftemp=sum(lambda_trad./(lambda_trad-mutemp))/M-1/c;
                fdtemp=sum(lambda_trad./((lambda_trad-mutemp).^2))/M;
                mutemp=mutemp-ftemp/fdtemp;
                ind=1;
                while mutemp>lambda_trad(m) | mutemp<lambda_trad(m-1),
                    mutemp=mutemp2-1/ind*ftemp/fdtemp;
                    ind=ind+1;
                end;
            end;
        else,
            mutemp=lambda_trad(m)/2;
            mutemp2=2*lambda_trad(m)/3;
            while abs((mutemp-mutemp2)./mutemp)>err,
                mutemp2=mutemp;
                ftemp=sum(lambda_trad./(lambda_trad-mutemp))/M-1/c;
                fdtemp=sum(lambda_trad./((lambda_trad-mutemp).^2))/M;
                mutemp=mutemp-ftemp/fdtemp;
                ind=1;
                while mutemp>lambda_trad(m),
                    mutemp=mutemp2-1/ind*ftemp/fdtemp;
                    ind=ind+1;
                end;
            end;
        end;
        muest(m)=mutemp;
    end;
else, % c>=1
    if min(abs(diff(lambda_trad(M-N+1:M))))==0
        fprintf('Error: Sample covariance matrix contains non-zero eigenvalues with multiplicity higher than one.\n')
        return
    end;
    for m=M-N+2:M,
        mutemp=(lambda_trad(m)+lambda_trad(m-1))/2;
        mutemp2=(2*lambda_trad(m)+lambda_trad(m-1))/3;
        while abs((mutemp-mutemp2)./mutemp)>err,
            mutemp2=mutemp;
            ftemp=sum(lambda_trad./(lambda_trad-mutemp))/M-1/c;
            fdtemp=sum(lambda_trad./((lambda_trad-mutemp).^2))/M;
            mutemp=mutemp-ftemp/fdtemp;
            ind=1;
            while mutemp>lambda_trad(m) | mutemp<lambda_trad(m-1),
                mutemp=mutemp2-1/ind*ftemp/fdtemp;
                ind=ind+1;
            end;
        end;
        muest(m)=mutemp;
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimators Construction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

theta=zeros(M,Mbar); % Weights of the proposed subspace method
for m=1:Mbar,
    index=(sum(Kvec(1:m-1))+1):sum(Kvec(1:m));
    indexc=[1:sum(Kvec(1:m-1)),(sum(Kvec(1:m))+1):M];

    % Proposed estimator for the m-th eigenvalue
    gamma_output(m)=N*sum(lambda_trad(index)-muest(index))/Kvec(m);
   
    % Eigenvectors construction
    phi=zeros(length(indexc),1);
    psi=zeros(length(index),1);

    for jj=1:length(index),
        if lambda_trad(index(jj))>0,
            psi(jj)=sum(lambda_trad(indexc)./(lambda_trad(index(jj))-lambda_trad(indexc))-muest(indexc)./(lambda_trad(index(jj))-muest(indexc)));
        else,
            psi(jj)=0;
        end;
    end;
    for jj=1:length(indexc),
        if lambda_trad(indexc(jj))>0,
            phi(jj)=sum(lambda_trad(index)./(lambda_trad(indexc(jj))-lambda_trad(index))-muest(index)./(lambda_trad(indexc(jj))-muest(index)));
        else,
            phi(jj)=0;
        end;
    end;

    theta(index,m)=1+psi;
    theta(indexc,m)=-phi;
    P_output(:,:,m)=E_trad*diag(theta(:,m))*E_trad';
        
end;

