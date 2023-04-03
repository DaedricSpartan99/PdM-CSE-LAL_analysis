function logL = log_likelihood_model(X, Parameters)

    %Y = uq_evalModel(Parameters.Model, X);
    %L = - (Y - Parameters.Measurement).^2 / Parameters.Discrepancy / 2. - log(2*pi*Parameters.Discrepancy) / 2.;

    Y = uq_evalModel(Parameters.ForwardModel, X);
    N = size(X,1);

    % TODO: vectorize, if it's slow it simulated a slow evaluation
    logL = zeros(N,1);
    for i = 1:N
        logL(i) = sum(-(Parameters.y - Y(i,1)).^2/ Parameters.Discrepancy /2. - 0.5*log(pi*Parameters.Discrepancy));
    end
end