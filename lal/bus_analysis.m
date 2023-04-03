function BayesianAnalysis = bus_analysis(BayesOpts)

    %% Bayesian inversion input options
    % BayesOpts.Prior:                              UQInput
    % BayesOpts.Discrepancy                         UQInput
    % BayesOpts.LogLikelihood:                      FunctionHandle
    % BayesOpts.Bus.logC:                           double
    % BayesOpts.Bus.p0:                             double, 0 < p0 < 0.5
    % BayesOpts.Bus.BatchSize:                      integer, > 0

    %% Bayesian inversion output results

    % BayesianAnalysis.Results.Evidence:            double
    % BayesianAnalysis.Results.PostSamples:         array K x M
    % BayesianAnalysis.Results.Subset:              Subset simulation Results struct
    % BayesianAnalysis.Results.Bus.LSF:             FunctionHandle
    % BayesianAnalysis.Results.Bus.PostSamples:     array K x (M+1) 
    % BayesianAnalysis.Opts:                        BayesOpts struct

    %% BuS composed input definition

    % Setup P random variable
    BusPriorOpts.Name = 'BusPrior';
    BusPriorOpts.Marginals(1).Name = 'P';
    BusPriorOpts.Marginals(1).Type = 'Uniform';
    BusPriorOpts.Marginals(1).Parameters = [0, 1];

    M = length(BayesOpts.Prior.Marginals);
    
    % Setup X random variable
    % TODO: manage correlation
    for i = 1:M
        %BusPriorOpts.Marginals(i+1) = BayesOpts.Prior.Options.Marginals(i);
        BusPriorOpts.Marginals(i+1).Type = BayesOpts.Prior.Marginals(i).Type;
        BusPriorOpts.Marginals(i+1).Parameters = BayesOpts.Prior.Marginals(i).Parameters;
    end

    %% Construct BuS Limit state function
    LSFOpts.mFile = 'limit_state_model';
    LSFOpts.isVectorized = true;
    LSFOpts.Parameters.logC = BayesOpts.Bus.logC;        
    LSFOpts.Parameters.LogLikelihood = BayesOpts.LogLikelihood;   % Set the surrogate model
    
    
    %% Setup and run the Subset simulation
    SSOpts.Type = 'Reliability';
    SSOpts.Method = 'Subset';

    if isfield(BayesOpts.Bus, 'p0')
        SSOpts.Subset.p0 = BayesOpts.Bus.p0;
    end

    if isfield(BayesOpts.Bus, 'BatchSize')
        SSOpts.Simulation.BatchSize = BayesOpts.Bus.BatchSize;
    end

    if isfield(BayesOpts.Bus, 'MaxSampleSize')
        SSOpts.Simulation.MaxSampleSize = BayesOpts.Bus.MaxSampleSize;
    end

    SSOpts.Model = uq_createModel(LSFOpts, '-private');
    SSOpts.Input = uq_createInput(BusPriorOpts, '-private');
    SSimAnalysis = uq_createAnalysis(SSOpts, '-private');

    %% Estimate reliability index

    LSFOpts.Parameters.ConfAlpha = 0.025;
    LSFOpts.mFile = 'limit_state_model_conf';

    % Run SuS for + failure probability
    LSFOpts.Parameters.ConfSign = '+';
    SSOpts.Model = uq_createModel(LSFOpts, '-private');
    SSimAnalysisPlus = uq_createAnalysis(SSOpts, '-private');

    % Run SuS for - failure probability
    LSFOpts.Parameters.ConfSign = '-';
    SSOpts.Model = uq_createModel(LSFOpts, '-private');
    SSimAnalysisMinus = uq_createAnalysis(SSOpts, '-private');

    % Compute Reliability index
    beta_0 = -norminv(SSimAnalysis.Results.Pf);
    beta_plus = -norminv(SSimAnalysisMinus.Results.Pf);
    beta_minus = -norminv(SSimAnalysisPlus.Results.Pf);

    
    %% Store results and opts
    BayesianAnalysis.Results.Evidence = exp(log(SSimAnalysis.Results.Pf) - BayesOpts.Bus.logC);
    BayesianAnalysis.Results.Subset = SSimAnalysis.Results;
    History = {SSimAnalysis.Results.History.X};

    % Reliability index
    BayesianAnalysis.Results.ReliabilityIndex = abs(beta_plus - beta_minus) / beta_0;

    % Quasi-posterior samples and evaluations
    BayesianAnalysis.Results.Bus.LSF = SSimAnalysis.Options.Model;
    BayesianAnalysis.Results.Bus.PostSamples = History{end}{end};

    % Options
    BayesianAnalysis.Opts = BayesOpts;
   
end