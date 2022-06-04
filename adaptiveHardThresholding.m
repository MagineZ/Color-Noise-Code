function [Xest, Topt, r] = adaptiveHardThresholding(Y, k, strategy)
    
    [U, fY, Vt] = svd(Y);
    fY = diag(fY);
    [p,n] = size(Y);
    gamma = min( p/n, n/p);
    fZ = createPseudoNoise(fY, k, strategy);
    Topt = computeOptThreshold(fZ, gamma);
    
    fY(fY<=Topt) = 0;
    n = length(fY);
    Xest = U * diag(fY) * Vt(:,1:n)';
    r = sum(fY>0);
    
end