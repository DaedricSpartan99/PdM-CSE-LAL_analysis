function logL = log_likelihood_model(X, Parameters)

    % compute discrepancy
    % TODO: more flexibility
    disc = X(:,end-Parameters.DiscrepancyDim+1:end); %uq_evalModel(Parameters.DiscrepancyModel, X(:,end-Parameters.DiscrepancyDim+1:end));

    Y = uq_evalModel(Parameters.ForwardModel, X(:,1:end-Parameters.DiscrepancyDim));
    N = size(X,1);
    Ny = length(Parameters.myData);

    logLY = zeros(N, Ny);
    
    for i = 1:Ny
        dY =  Y(:,Parameters.myData(i).MOMap) - repmat(Parameters.myData(i).y, size(Y,1), 1);
        dYSY = sum(dY.^2, 2) ./ disc(:,i);
        logLY(:,i) = - dYSY ./ 2. - size(dY,2)*0.5*log(2*pi*disc(:,i));
    end

    logL = sum(logLY, 2) .* Parameters.Amplification;
end