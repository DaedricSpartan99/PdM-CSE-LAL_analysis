function logL = log_likelihood_model(X, Parameters)

    % compute discrepancy
    disc = var(Parameters.y);
    Y = uq_evalModel(Parameters.ForwardModel, X);
    N = size(X,1);

    % TODO: vectorize, if it's slow it simulated a slow evaluation
    logL = zeros(N,1);
    for i = 1:N
        logL(i) = sum(-(Parameters.y - Y(i,1)).^2/ disc /2. - 0.5*log(pi*disc));
    end
end