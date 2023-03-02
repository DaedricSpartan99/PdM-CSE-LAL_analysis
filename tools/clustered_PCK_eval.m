function [Y, varY] = clustered_PCK_eval(X, Parameters)
    
    labels = clustered_PCK_classify(Parameters, X);

    % Group by labels and evaluate
    unique_labels = unique(labels);
    Y = zeros(size(X,1),Parameters.dimY);
    varY = zeros(size(X,1),Parameters.dimY);

    for i=1:length(unique_labels)

        % Restrict to label
        label = unique_labels(i);
        metaModel = Parameters.models{label};
        X_subset = X(labels == label,:);

        % Evaluate to specific label
        [Y_sub, varY_sub] = uq_evalModel(metaModel, X_subset);

        % Assign to correct position
        Y(labels == label) = Y_sub;
        varY(labels == label) = varY_sub;
    end
end