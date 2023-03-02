function labels = clustered_PCK_classify(CPCK, X)
    
    z = (X - CPCK.norm.meanX) ./ CPCK.norm.stdX; 
    distances = pdist2(z, CPCK.norm.Z(:,1:CPCK.dimX),'euc');

    [~, idx] = min(distances, [], 2);
    labels = CPCK.labels(idx);
end