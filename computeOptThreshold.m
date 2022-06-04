function mid = computeOptThreshold(fZ, gamma)

low = max(fZ);
high = low + 2.0;

while F(high, fZ, gamma) < -4
    low = high;
    high = 2*high;
end

% F is increasing, do binary search:
eps = 10e-6;
while high-low > eps
    mid = (high+low)/2;
    if F(mid, fZ, gamma) < -4
        low = mid;
    else
        high = mid;
    end
end
end

function     yy =   F(y, fZ, gamma)
d = D(y, fZ, gamma);
dd = Dd(y, fZ, gamma);
yy  = y * dd / d;
end

function yy =  Phi(y, fZ)
    phi = y./(y.^2 - fZ.^2);
    yy = mean(phi);
end

function yy = Phid(y, fZ)
    phid = -(y.^2+fZ.^2)./(y.^2-fZ.^2).^2;
    yy = mean(phid);
end

function yy = D(y, fZ, gamma)
    phi = Phi(y, fZ);
    yy =  phi * (gamma*phi + (1-gamma)./y); 
end

function yy = Dd(y, fZ, gamma)
    phi = Phi(y, fZ);
    phid = Phid(y, fZ);
    yy =  phid * (gamma*phi + (1-gamma)./y) + phi * (gamma*phid - (1-gamma)./y^2);
end