function [Z,fT] = Noise(p,n,name,Q)
 X = randn(p,n)/sqrt(n);
 switch name 
    case 'Marcenko-Pastur'
        fT = ones(1,p); 
        fT = diag(sqrt(fT));
        Z = Q*fT*Q'*X;
    case 'Mix2'
        fT = ones(1,p);
        fT(1:floor(p/2))=10.0;
        fT = diag(sqrt(fT));
        Z = Q*fT*Q'*X;
    case 'Unif[1,10]'
        fT = ones(1,p);  
        for l = 1:p
           fT(l) = fT(l) + 9*l/p;
        end
        fT = diag(sqrt(fT));
        Z = Q*fT*Q'*X;
    case 'PaddedIdentity'
        Z =  [eye(p); zeros(n-p,p)] ;
    
    case 'Chi10'
        fT = randn(floor(sqrt(p)),10);
        fT = 4*mean(fT.^2, 2);
        fT = [fT;ones(p-length(fT),1)];
        fT = diag(sqrt(fT));
        Z = Q*fT*Q'*X;
        
    case 'Fisher3n'
        T = randn(3*p, p)/sqrt(3*p);
        [~, fT, ~] = svd(T);
        fT = 1./fT;
        fT = diag(sqrt(fT));
        fT = diag(fT);
        Z = Q*fT*Q'*X;
     case 'Triangle'
         
        a = 1;b = 0.4;c = 0.4; %d = 0.4; e = 0.4; f = 0.3; g = 0.3;
        Sigma = diag(a*ones(1,p)) + diag(b*ones(1,p-1),1) + diag(c*ones(1,p-1),-1);%+ diag(d*ones(1,p-2),2)+diag(e*ones(1,p-2),-2);
        %Sigma = diag(a*ones(1,p)) + diag(b*ones(1,p-1),1) + diag(c*ones(1,p-1),-1)+ diag(d*ones(1,p-2),2)+diag(e*ones(1,p-2),-2)+diag(f*ones(1,p-3),3)+diag(g*ones(1,p-3),-3);
        [U,D] = eig(Sigma);
        fT = U*D.^(1/2);
        Z = fT*X;
        
        
 end
 
    



