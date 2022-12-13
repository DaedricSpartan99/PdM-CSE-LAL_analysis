function L = log_likelihood_model(X, Parameters)

    Y = uq_evalModel(Parameters.Model, X);
    L = - (Y - Parameters.Measurement).^2 / Parameters.Discrepancy / 2. - log(2*pi*Parameters.Discrepancy) / 2.;
end