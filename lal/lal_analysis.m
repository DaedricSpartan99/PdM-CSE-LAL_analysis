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

        X = uq_getSample(Opts.Prior, Opts.ExpDesign.InitEval, 'LHS');

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
        logLhist = histogram(ax3, logL);
        title(ax3, 'Experimental design emplacement')
        xlabel('Log-likelihood')

        ax4 = nexttile;
        W = pca(X);
        T = X * W(:,1:2);
        pca_scatter = scatter(ax4, T(:,1), T(:,2), 20, logL, 'Filled')
        pca_colorbar = colorbar(ax4)
        title('Experimental design PCA')
        xlabel('x1')
        ylabel('x2')
        
        drawnow
    end

    %post_input = Opts.Prior;
    
    % Begin iterations
    for i = 1:Opts.MaximumEvaluations

        % Get clusters

        % TODO: cross validate minpts and quantile, maximize number
        dbscan_minpts = Opts.dbscanMinpts;
        dbscan_kD = pdist2(X,X,'euc','Smallest',dbscan_minpts);
        
        dbscan_kD_sorted = sort(dbscan_kD(end,:));
        dbscan_epsilon = quantile(dbscan_kD_sorted,Opts.dbscanQuantile);

        dbscan_labels = dbscan(X,dbscan_epsilon,dbscan_minpts);

        unique_labels = sort(unique(dbscan_labels));
        unique_labels = unique_labels(2:end); % Skip outliers
        nb_labels = length(unique_labels);

        sprintf("Clusters found: %d", nb_labels)

        % Collect best options
        X_star = zeros(nb_labels, size(X,2));
        cost_star = zeros(nb_labels,1);
        pcks = cell(nb_labels,1);

        for label_index = 1:nb_labels

            % Restrict experimental design to cluster
            label = unique_labels(label_index);
            X_label = X(dbscan_labels == label,:);
            logL_label = logL(dbscan_labels == label);
            
            % Construct a PC-Kriging surrogate of the log-likelihood
            PCKOpts = Opts.PCK;
            PCKOpts.Type = 'Metamodel';
            PCKOpts.MetaType = 'PCK';
            PCKOpts.Mode = 'optimal';  
            %PCKOpts.FullModel = Opts.LogLikelihood;
            PCKOpts.Input = Opts.Prior; 
            PCKOpts.ExpDesign.X = X_label;
            PCKOpts.ExpDesign.Y = logL_label;
            
            PCKOpts.ValidationSet.X = Opts.Validation.PostSamples;
            PCKOpts.ValidationSet.Y = Opts.Validation.PostLogLikelihood;
    
            logL_PCK = uq_createModel(PCKOpts, '-private');
            pcks{label_index} = logL_PCK;
    
            sprintf("Iteration number: %d, Label = %d", i, label)
            sprintf("PCK LOO error: %g", logL_PCK.Error.LOO)
            sprintf("PCK Validation error: %g", logL_PCK.Error.Val)
            
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
            [opt_cost, opt_index] = min(cost_LSF);
            xopt = BusAnalysis.Results.PostSamples(opt_index, :);
            
            X_star(label_index,:) = xopt;
            cost_star(label_index) = opt_cost;
        end

        sprintf("All clustering steps done, applying merging strategy")

        % Merge optimal points
        % TODO: improve strategy by considering all samples
        dbscan_weights = zeros(nb_labels,1);

        for label_index =1:nb_labels
            dbscan_weights(label_index) = sum(dbscan_labels == unique_labels(label_index)) * 1.0 / size(X,1);
        end

        [~,xopt_index] = min(dbscan_weights .* cost_star);
        xopt = X_star(xopt_index,:);

        % Add to experimental design
        X = [X; xopt];
        logL = [logL; Opts.LogLikelihood(xopt) ];

        % Update plot
        if Opts.PlotLogLikelihood

            % Get optimal label
            val_errs = zeros(nb_labels);
            for label_index = 1:nb_labels
                val_errs(label_index) = pcks{label_index}.Error.Val;
            end
            [~, opt_label_index] = min(val_errs);
            %opt_label = unique_labels(opt_label_index);

            % Create a PCK with full set
            PCKOpts = Opts.PCK;
            PCKOpts.Type = 'Metamodel';
            PCKOpts.MetaType = 'PCK';
            PCKOpts.Mode = 'optimal';  
            %PCKOpts.FullModel = Opts.LogLikelihood;
            PCKOpts.Input = Opts.Prior; 
            PCKOpts.ExpDesign.X = X(dbscan_labels ~= -1, :);
            PCKOpts.ExpDesign.Y = logL(dbscan_labels ~= -1);
            
            PCKOpts.ValidationSet.X = Opts.Validation.PostSamples;
            PCKOpts.ValidationSet.Y = Opts.Validation.PostLogLikelihood;
    
            %logL_PCK = uq_createModel(PCKOpts, '-private');
            logL_PCK = pcks{opt_label_index};

            sprintf("PCK LOO error: %g", logL_PCK.Error.LOO)
            sprintf("PCK Validation error: %g", logL_PCK.Error.Val)

            set(post_valid_plot, 'XData', Opts.Validation.PostLogLikelihood, 'YData', uq_evalModel(logL_PCK, Opts.Validation.PostSamples));
            set(prior_valid_plot, 'XData', Opts.Validation.PriorLogLikelihood, 'YData', uq_evalModel(logL_PCK, Opts.Validation.PriorSamples));
            set(logLhist, 'Data', logL);

            W = pca(X);
            T = X * W(:,1:2);
            set(pca_scatter, 'XData',T(:,1), 'YData', T(:,2), "CData", logL)
            set(pca_colorbar, 'Limits', [min(logL), max(logL)])
            %pca_scatter = scatter(T(:,1), T(:,2), "ColorVariable", logL, 'Filled')

            drawnow
        end
    end

    % Store results
    LALAnalysis.ExpDesign.X = X;
    LALAnalysis.ExpDesign.LogLikelihood = logL;
    LALAnalysis.BusAnalysis = BusAnalysis;
    LALAnalysis.Opts = Opts;
end