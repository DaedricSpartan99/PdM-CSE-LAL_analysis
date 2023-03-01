function labels = clustered_PCK_classify(CPCK, X)
    
    z = (X - CPCK.norm.meanX) ./ CPCK.norm.stdX; 
    distances = pdist2(z, cpck.norm.Z,'euc','Smallest',CPCK.Options.MinPts);

    [~, idx] = min(distances, [], 2);
    labels = CPCK.labels(idx);
end