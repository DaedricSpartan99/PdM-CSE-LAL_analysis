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

    % Initialize output following initial guesses
    if isfield(Opts.ExpDesign, 'InitEval')

        X = uq_getSample(Opts.Prior, Opts.ExpDesign.InitEval, 'LHS'); % 'LHS'

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
        tiledlayout(2,2)

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

        ax3 = nexttile;
        logLhist = histogram(ax3, logL, 12);
        title(ax3, 'Experimental design emplacement')
        xlabel('Log-likelihood')

        ax4 = nexttile;
        W = pca(X);
        T = X * W(:,1:2);
        pca_scatter = scatter(ax4, T(:,1), T(:,2), 20, logL, 'Filled')
        pca_colorbar = colorbar(ax4)
        title(ax4, 'Experimental design PCA')
        xlabel(ax4, 'x1')
        ylabel(ax4, 'x2')
        
        drawnow
    end

    if isfield(Opts, 'StoreBusResults') && Opts.StoreBusResults
        LALAnalysis.lsfEvaluations = cell(Opts.MaximumEvaluations,1);
    end

    %post_input = Opts.Prior;
    
    % Begin iterations
    for i = 1:Opts.MaximumEvaluations

        % Address instabilities in the experimental design (0.05 quantile)
        if isfield(Opts, 'cleanQuantile')   
            in_logL_mask = logL > quantile(logL,Opts.cleanQuantile);
        else
            in_logL_mask = ones(size(logL,1),1);
        end

        %X_cleaned = X(in_logL_mask,:);
        %logL_cleaned = logL(in_logL_mask);
            
        % Construct a PC-Kriging surrogate of the log-likelihood
        PCKOpts = Opts.PCK;
        PCKOpts.Type = 'Metamodel';
        PCKOpts.MetaType = 'PCK';
        PCKOpts.Mode = 'optimal';  
        %PCKOpts.FullModel = Opts.LogLikelihood;
        PCKOpts.Input = Opts.Prior; 
        PCKOpts.ExpDesign.X = X;
        PCKOpts.ExpDesign.Y = logL;
        
        if isfield(Opts, 'Validation')
            PCKOpts.ValidationSet.X = Opts.Validation.PostSamples;
            PCKOpts.ValidationSet.Y = Opts.Validation.PostLogLikelihood;
        end

        logL_PCK = uq_createModel(PCKOpts, '-private');

        sprintf("Iteration number: %d", i)
        sprintf("PCK LOO error: %g", logL_PCK.Error.LOO)

        if isfield(Opts, 'Validation')
            sprintf("PCK Validation error: %g", logL_PCK.Error.Val)
        end
        
        % TODO: Determine optimal c = 1 / max(L)

        % Execute Bayesian Analysis in Bus framework
        BayesOpts.Prior = Opts.Prior;
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

        % Store result as history
        LALAnalysis.OptPoints(i).X = xopt;
        LALAnalysis.OptPoints(i).logL = logL(end);
        LALAnalysis.OptPoints(i).lsf = cost_LSF(opt_index);

        if isfield(Opts, 'StoreBusResults') && Opts.StoreBusResults
            % Store in results
            LALAnalysis.BusAnalysis(i) = BusAnalysis;
            % Store evaluations
            LALAnalysis.lsfEvaluations{i} = cost_LSF;
        end

        %if abs(logL(end) - min(logL)) < eps
        %    sprintf("Found unconvenient point, logL = %g", logL(end))
        %end

        % Update plot
        if Opts.PlotLogLikelihood
            set(post_valid_plot, 'XData', Opts.Validation.PostLogLikelihood, 'YData', uq_evalModel(logL_PCK, Opts.Validation.PostSamples));
            set(prior_valid_plot, 'XData', Opts.Validation.PriorLogLikelihood, 'YData', uq_evalModel(logL_PCK, Opts.Validation.PriorSamples));
            set(logLhist, 'Data', logL);

            W = pca(X_cleaned);
            T = X_cleaned * W(:,1:2);
            set(pca_scatter, 'XData',T(:,1), 'YData', T(:,2), "CData", logL_cleaned)
            set(pca_colorbar, 'Limits', [min(logL_cleaned), max(logL_cleaned)])
            %pca_scatter = scatter(T(:,1), T(:,2), "ColorVariable", logL, 'Filled')

            drawnow
        end
    end

    % Store results
    LALAnalysis.ExpDesign.X = X;
    LALAnalysis.ExpDesign.LogLikelihood = logL;
    LALAnalysis.Opts = Opts;
end