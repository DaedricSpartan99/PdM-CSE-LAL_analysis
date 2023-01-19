function LALAnalysis = lal_analysis(Opts)

    %% Input Options

    % Opts.MaximumEvaluations:      int, > 0
    % Opts.ExpDesign.X:             array N x M
    % Opts.ExpDesign.LogLikelihood: array N x 1
    % Opts.ExpDesign.InitEval       int       
    % Opts.LogLikelihood:           UQModel
    % Opts.Prior:                   UQInput
    % Opts.Discrepancy              UQInput
    % Opts.Bus.logC:                double
    % Opts.Bus.p0:                  double, 0 < p0 < 0.5
    % Opts.Bus.BatchSize:           int, > 0
    % Opts.Bus.MaxSampleSize        int, > 0
    % Opts.PCE.MinDegree:           int, > 0
    % Opts.PCE.MaxDegree:           int, > 0
    % Opts.ExpDesign.FilterZeros    logical, filter experimental design

    %% Output fields

    % LALAnalysis.ExpDesign.X:              enriched design, array N_out x M
    % LALAnalysis.ExpDesign.LogLikelihood:  enriched design, array N_out x 1
    % LALAnalysis.BusAnalysis:              BusAnalysis struct
    % LALAnalysis.Opts:                     LALAnalysis options struct

    %% Execution

    % Create joint input
    JointPriorOpts.Name = strcat('Joint', Opts.Prior.Name);
    JointPriorOpts.Marginals = Opts.Prior.Marginals;

    M = length(JointPriorOpts.Marginals);

    if isfield(Opts, 'Discrepancy')
        for i = 1:length(Opts.Discrepancy)
            JointPriorOpts.Marginals(M+i) = Opts.Discrepancy(i).Prior.Marginals;
        end
    end

    JointPriorOpts.Marginals = rmfield(JointPriorOpts.Marginals, 'Moments');

    JointPrior = uq_createInput(JointPriorOpts, '-private');


    % Initialize output following initial guesses
    if isfield(Opts.ExpDesign, 'InitEval')

        X = uq_getSample(JointPrior, Opts.ExpDesign.InitEval);

        logL = uq_evalModel(Opts.LogLikelihood, X);
    else
        X = Opts.ExpDesign.X;
        logL = Opts.ExpDesign.LogLikelihood;
    end

    if ~isfield(Opts, 'PlotLogLikelihood')
        Opts.PlotLogLikelihood = false;
    end

    % plot setup
    if Opts.PlotLogLikelihood
        figure
        title('Log-likelihood convergence')
        ylabel('Log-Likelihood')
        xlabel('Marginal target X')

        X5_K = X(:,Opts.PlotTarget);
        logL_K = logL;
        
        hold on
        sp5 = scatter(X(:,Opts.PlotTarget), logL);
        pp5 = plot(X5_K, logL_K);
        pl5 = plot(X5_K, logL_K);
        hold off
        legend('Experimental design', 'PCK', 'Real Log-Likelihood')
        
        drawnow
    end
    
    % Begin iterations
    for i = 1:Opts.MaximumEvaluations

        % Apply experimental design filtering
        if isfield(Opts.ExpDesign, 'FilterZeros') && Opts.ExpDesign.FilterZeros
            indexes = logL > log(eps);
            X = X(indexes,:);
            logL = logL(indexes);
        end
    
        % Construct a PC-Kriging surrogate of the log-likelihood
        PCKOpts.Type = 'Metamodel';
        PCKOpts.MetaType = 'PCK';
        PCKOpts.Mode = 'optimal';
        PCKOpts.FullModel = Opts.LogLikelihood;
        PCKOpts.Input = JointPrior; 
        PCKOpts.PCE.Degree = Opts.PCE.MinDegree:Opts.PCE.MaxDegree;
        PCKOpts.PCE.Method = 'LARS';
        PCKOpts.ExpDesign.X = X;
        PCKOpts.ExpDesign.Y = logL;
        PCKOpts.Kriging.Optim.Method = 'CMAES';
        PCKOpts.Kriging.Corr.Family = 'Gaussian';
        %PCKOpts.Display = 'verbose';
    
        logL_PCK = uq_createModel(PCKOpts, '-private');
        
        % TODO: Determine optimal c = 1 / max(L)

        % Execute Bayesian Analysis in Bus framework
        BayesOpts.Prior = JointPrior;
        BayesOpts.Bus = Opts.Bus;
        BayesOpts.LogLikelihood = logL_PCK;

        % Adaptively determine constant Bus.logC
        % TODO: better algorithm
        if ~isfield(Opts.Bus, 'logC')

            % Default strategy
            if ~isfield(BayesOpts.Bus, 'CStrategy')
                BayesOpts.Bus.CStrategy = 'max';
            end
           
            % Take specified strategy
            if BayesOpts.Bus.CStrategy == 'max';
                BayesOpts.Bus.logC = -max(logL);
            else if Opts.Bus.CStrategy == 'latest';
                BayesOpts.Bus.logC = -logL(end);
            end

            sprintf("Taking constant logC: %g", BayesOpts.Bus.logC);
        end
    
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

        if Opts.PlotLogLikelihood
            n_plot = 1000;

            %X_plot = sortrows(uq_getSample(JointPrior, n_plot), Opts.PlotTarget);

            % Use BUS posterior
            X_plot = sortrows(BusAnalysis.Results.PostSamples(1:n_plot, :), Opts.PlotTarget);

            % Evaluate responses
            logL_K = uq_evalModel(logL_PCK, X_plot);
            logL_real = uq_evalModel(Opts.LogLikelihood, X_plot);

            set(sp5, 'XData', X(:,Opts.PlotTarget), 'YData', logL);
            set(pp5, 'XData', X_plot(:,Opts.PlotTarget), 'YData', logL_K);
            set(pl5, 'XData', X_plot(:,Opts.PlotTarget), 'YData', logL_real);

            drawnow
        end
    end

    % Store results
    LALAnalysis.ExpDesign.X = X;
    LALAnalysis.ExpDesign.LogLikelihood = logL;
    LALAnalysis.BusAnalysis = BusAnalysis;
    LALAnalysis.Opts = Opts;
end