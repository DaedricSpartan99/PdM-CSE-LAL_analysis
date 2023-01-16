function Sigma = discrepancy_model(params)
    Sigma = zeros(size(params,1), size(params,2), size(params,2));

    % simple model
    for i = 1:size(params,1)
        Sigma(i,:,:) = diag(params(i));
    end
end