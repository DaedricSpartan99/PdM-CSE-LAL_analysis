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
    %JointPriorOpts.Name = strcat('Joint', Opts.Prior.Name);
    %JointPriorOpts.Marginals = Opts.Prior.Marginals;

    %M = length(JointPriorOpts.Marginals);

    %if isfield(Opts, 'Discrepancy')
    %    for i = 1:length(Opts.Discrepancy)
    %        JointPriorOpts.Marginals(M+i) = Opts.Discrepancy(i).Prior.Marginals;
    %    end
    %end

    %JointPriorOpts.Marginals = rmfield(JointPriorOpts.Marginals, 'Moments');

    %JointPrior = uq_createInput(JointPriorOpts, '-private');


    % Initialize output following initial guesses
    if isfield(Opts.ExpDesign, 'InitEval')

        X = uq_getSample(Opts.Prior, Opts.ExpDesign.InitEval);

        logL = Opts.LogLikelihood(X);
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
        tiledlayout(1,2)

        check_interval = [min(Opts.Validation.PostLogLikelihood), max(Opts.Validation.PostLogLikelihood)];

        ax1 = nexttile;
        hold on
        plot(ax1, check_interval, check_interval);
        post_valid_plot = scatter(ax1, Opts.Validation.PostLogLikelihood, Opts.Validation.PostLogLikelihood);
        hold off
        title(ax1, 'Posterior samples')
        ylabel(ax1, 'Surrogate Log-Likelihood')
        xlabel(ax1, 'Real Log-Likelihood')
        xlim(check_interval)
        ylim(check_interval)

        check_interval = [min(Opts.Validation.PriorLogLikelihood), max(Opts.Validation.PriorLogLikelihood)];

        ax2 = nexttile;
        hold on
        plot(ax2, check_interval , check_interval);
        prior_valid_plot = scatter(ax2, Opts.Validation.PriorLogLikelihood, Opts.Validation.PriorLogLikelihood);
        hold off
        title(ax2, 'Prior samples')
        ylabel(ax2, 'Surrogate Log-Likelihood')
        xlabel(ax2, 'Real Log-Likelihood')
        xlim(check_interval)
        ylim(check_interval)
        
        drawnow
    end

    %post_input = Opts.Prior;
    
    % Begin iterations
    for i = 1:Opts.MaximumEvaluations
    
        % Construct a PC-Kriging surrogate of the log-likelihood
        PCKOpts = Opts.PCK;
        PCKOpts.Type = 'Metamodel';
        PCKOpts.MetaType = 'PCK';
        %PCKOpts.Mode = 'optimal'; %'sequential'; 
        %PCKOpts.FullModel = Opts.LogLikelihood;
        PCKOpts.Input = Opts.Prior; 
        PCKOpts.ExpDesign.X = X;
        PCKOpts.ExpDesign.Y = logL;
        
        PCKOpts.ValidationSet.X = Opts.Validation.PostSamples;
        PCKOpts.ValidationSet.Y = Opts.Validation.PostLogLikelihood;

        logL_PCK = uq_createModel(PCKOpts, '-private');

        sprintf("Iteration number: %d", i)
        sprintf("PCK LOO error: %g", logL_PCK.Error.LOO)
        sprintf("PCK Validation error: %g", logL_PCK.Error.Val)
        
        % TODO: Determine optimal c = 1 / max(L)

        % Execute Bayesian Analysis in Bus framework
        BayesOpts.Prior = Opts.Prior;
        BayesOpts.Bus = Opts.Bus;
        BayesOpts.LogLikelihood = logL_PCK; %;

        % Adaptively determine constant Bus.logC
        % TODO: better algorithm
        if ~isfield(Opts.Bus, 'logC')

            % Default strategy
            if ~isfield(BayesOpts.Bus, 'CStrategy')
                BayesOpts.Bus.CStrategy = 'max';
            end
           
            % Take specified strategy
            if BayesOpts.Bus.CStrategy == 'max'
                BayesOpts.Bus.logC = -max(logL);
            elseif Opts.Bus.CStrategy == 'latest'
                BayesOpts.Bus.logC = -logL(end);
            end

            sprintf("Taking constant logC: %g", BayesOpts.Bus.logC)
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
        logL = [logL; Opts.LogLikelihood(xopt) ];

        % Update plot
        if Opts.PlotLogLikelihood
            set(post_valid_plot, 'XData', Opts.Validation.PostLogLikelihood, 'YData', uq_evalModel(logL_PCK, Opts.Validation.PostSamples));
            set(prior_valid_plot, 'XData', Opts.Validation.PriorLogLikelihood, 'YData', uq_evalModel(logL_PCK, Opts.Validation.PriorSamples));

            drawnow
        end

        % Posterior handle evaluation
        %PostOpts.marginals(1).Type = 'posterior';

        %post_input = uq_createInput(PostOpts, '-private');
    end

    % Store results
    LALAnalysis.ExpDesign.X = X;
    LALAnalysis.ExpDesign.LogLikelihood = logL;
    LALAnalysis.BusAnalysis = BusAnalysis;
    LALAnalysis.Opts = Opts;
end