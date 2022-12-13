function LALAnalysis = lal_analysis(Opts)

    %% Input Options

    % Opts.MaximumEvaluations:      int, > 0
    % Opts.ExpDesign.X:             array N x M
    % Opts.ExpDesign.LogLikelihood: array N x 1
    % Opts.LogLikelihood:           UQModel
    % Opts.Prior:                   UQInput
    % Opts.Bus.c:                   positive double
    % Opts.Bus.p0:                  double, 0 < p0 < 0.5
    % Opts.Bus.BatchSize:           int, > 0
    % Opts.Bus.MaxSampleSize        int, > 0
    % Opts.PCE.MinDegree:           int, > 0
    % Opts.PCE.MaxDegree:           int, > 0
    % Opts.PCE.Method               string, PCE coeff, 'LARS', 'OLS',...

    %% Output fields

    % LALAnalysis.ExpDesign.X:              enriched design, array N_out x M
    % LALAnalysis.ExpDesign.LogLikelihood:  enriched design, array N_out x 1 
    % LALAnalysis.BusAnalysis:              BusAnalysis struct
    % LALAnalysis.Opts:                     LALAnalysis options struct

    %% Execution

    % Initialize output following initial guesses
    X = Opts.ExpDesign.X;
    logL = Opts.ExpDesign.LogLikelihood;
    
    % Begin iterations
    for i = 1:Opts.MaximumEvaluations
    
        % Construct a PC-Kriging surrogate of the log-likelihood
        PCKOpts.Type = 'Metamodel';
        PCKOpts.MetaType = 'PCK';
        PCKOpts.Mode = 'sequential';
        PCKOpts.FullModel = Opts.LogLikelihood;
        PCKOpts.PCE.Degree = Opts.PCE.MinDegree:2:Opts.PCE.MaxDegree;
        %PCKOpts.PCE.Method = 'LARS';
        PCKOpts.ExpDesign.X = X;
        PCKOpts.ExpDesign.Y = logL;
        PCKOpts.Kriging.Corr.Family = 'Gaussian';
    
        logL_PCK = uq_createModel(PCKOpts, '-private');
        
        % TODO: Determine optimal c = 1 / max(L)

        % Execute Bayesian Analysis in Bus framework
        BayesOpts.Prior = Opts.Prior;
        BayesOpts.Bus = Opts.Bus;
        BayesOpts.LogLikelihood = logL_PCK;
    
        BusAnalysis = bus_analysis(BayesOpts);
    
        % evaluate U-function on the limit state function
        % Idea: maximize misclassification probability
        % TODO: use a UQLab module for efficiency, but now not necessary
        [mean_post_LSF, var_post_LSF] = uq_evalModel(BusAnalysis.Results.Bus.LSF, BusAnalysis.Results.Bus.PostSamples);
        cost_LSF = abs(mean_post_LSF) ./ sqrt(var_post_LSF);
        [~, opt_index] = min(cost_LSF);
        xopt = BusAnalysis.Results.PostSamples(opt_index, :);
    
        % Add to experimental design
        X = [X; xopt];
        logL = [logL; uq_evalModel(Opts.LogLikelihood, xopt) ];
    end

    % Store results
    LALAnalysis.ExpDesign.X = X;
    LALAnalysis.ExpDesign.LogLikelihood = logL;
    LALAnalysis.BusAnalysis = BusAnalysis;
    LALAnalysis.Opts = Opts;
end