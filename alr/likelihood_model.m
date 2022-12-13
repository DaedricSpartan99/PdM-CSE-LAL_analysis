function L = likelihood_model(X, Parameters)

    Y = uq_evalModel(Parameters.Model, X);
    L = exp(- (Y - Parameters.Measurement).^2 / Parameters.Discrepancy / 2) / sqrt(2*pi*Parameters.Discrepancy);
end