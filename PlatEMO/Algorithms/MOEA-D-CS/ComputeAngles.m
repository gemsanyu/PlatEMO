function angles = ComputeAngles(A,B)
%COMPUTEANGLE Summary of this function goes here
%   Detailed explanation goes here
    normA = sqrt(sum(A.^2,2));
    normB = sqrt(sum(B.^2,2));    
    u = A*transpose(B);
    v = normA*transpose(normB);
    
    cosAngles = max(min(u./v,1),-1)
    angles = real(acos(cosAngles));
end

