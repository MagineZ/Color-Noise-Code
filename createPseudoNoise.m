function fZ =  createPseudoNoise(fY, k, strategy)

fZ = sort(fY,'descend');
p = length(fZ);
if k >= p
    error('k too large. procedure requires k < min(n,p)');
end

if k > 0
    if strategy == '0'
        fZ(1:k) = 0;
    elseif strategy == 'w'
        fZ(1:k) = fZ(k+1);
    elseif strategy == 'i'
        if 2*k+1 >= p
            error(ValueError('k too large. imputation requires 2*k+1 < min(n,p)'));
        end
        diff = fZ(k+1) - fZ(2*k+1);
        for l = 1:k
            a = (1 - ((l-1)/k)^(2/3)) / (2^(2/3)-1);
            fZ(l) = fZ(k+1) + a*diff;
        end
    else
        error( 'unknown strategy');
    end
end
