function [Y, varY] = clustered_PCK_eval(CPCK, X)
    
    label = clustered_PCK_classify(CPCK, X);
    Z = (X - CPCK.norm.meanX) ./ CPCK.norm.stdX; 

    metaModel = CPCK.models{label};
    [Y, varY] = uq_evalModel(metaModel, Z);
end