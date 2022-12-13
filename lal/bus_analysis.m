function BayesianAnalysis = bus_analysis(BayesOpts)

    %% Bayesian inversion input options
    % BayesOpts.Prior:                              UQInput
    % BayesOpts.LogLikelihood:                      FunctionHandle
    % BayesOpts.Bus.c:                              double, > 0
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
    
    % Setup X random variable
    % TODO: manage correlation
    for i = 1:length(BayesOpts.Prior.Marginals)
        %BusPriorOpts.Marginals(i+1) = BayesOpts.Prior.Options.Marginals(i);
        BusPriorOpts.Marginals(i+1).Type = BayesOpts.Prior.Marginals(i).Type;
        BusPriorOpts.Marginals(i+1).Parameters = BayesOpts.Prior.Marginals(i).Parameters;
        %BusPriorOpts.Marginals(i+1).Moments = BayesOpts.Prior.Marginals(i).Moments;
    end

    %% Construct BuS Limit state function
    LSFOpts.mFile = 'limit_state_model';
    LSFOpts.isVectorized = true;
    LSFOpts.Parameters.c = BayesOpts.Bus.c;        
    LSFOpts.Parameters.LogLikelihood = BayesOpts.LogLikelihood;   % Set the surrogate model
    
    
    %% Setup and run the Subset simulation
    SSOpts.Type = 'Reliability';
    SSOpts.Method = 'Subset';
    SSOpts.Subset.p0 = BayesOpts.Bus.p0;
    SSOpts.Simulation.BatchSize = BayesOpts.Bus.BatchSize;
    SSOpts.Simulation.MaxSampleSize = BayesOpts.Bus.MaxSampleSize;
    SSOpts.Model = uq_createModel(LSFOpts, '-private');
    SSOpts.Input = uq_createInput(BusPriorOpts, '-private');
    SSimAnalysis = uq_createAnalysis(SSOpts, '-private');
    
    %% Store results and opts
    BayesianAnalysis.Results.Evidence = SSimAnalysis.Results.Pf / BayesOpts.Bus.c;
    BayesianAnalysis.Results.Subset = SSimAnalysis.Results;
    History = {SSimAnalysis.Results.History.X};
    BayesianAnalysis.Results.PostSamples = History{end}{end}(:,2:end);

    BayesianAnalysis.Results.Bus.LSF = SSOpts.Model;
    BayesianAnalysis.Results.Bus.PostSamples = History{end}{end};

    BayesianAnalysis.Opts = BayesOpts;
   
end