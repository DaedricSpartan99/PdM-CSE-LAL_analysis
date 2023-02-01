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

    if ~isfield(Opts, 'dbscanQuantile')
        Opts.dbscanQuantile = 'auto';
    end

    if ~isfield(Opts, 'lsfEvaluations')
        Opts.lsfEvaluations = 10;
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
        dbscan_kD = pdist2([X,logL],[X,logL],'euc','Smallest',dbscan_minpts);
        
        dbscan_kD_sorted = sort(dbscan_kD(end,:));

        if Opts.dbscanQuantile == 'auto'
            
            sprintf("dbscanQuantile not specified, optimizing it")

            dbscan_midpoint = 0.5;
            dbscan_extension = 0.5;
            dbscan_grid_size = 20;

            for reduction = 1:4

                quantile_range = [max(dbscan_midpoint - dbscan_extension,0),min(dbscan_midpoint + dbscan_extension,1)];

                quantiles = linspace(quantile_range(1), quantile_range(2),dbscan_grid_size);
                nb_labels = zeros(dbscan_grid_size,1);

                for iq = 1:dbscan_grid_size
                    
                    dbscan_epsilon = quantile(dbscan_kD_sorted,quantiles(iq));
                    dbscan_labels = dbscan([X,logL],dbscan_epsilon,dbscan_minpts);
                    unique_labels = sort(unique(dbscan_labels));
                    unique_labels = unique_labels(2:end); % Skip outliers
                    nb_labels(iq) = length(unique_labels);
                end

                if length(unique(nb_labels)) == 1
                    break
                end

                [~, midpoint_index] = max(nb_labels);
                dbscan_midpoint = quantiles(midpoint_index);
                dbscan_extension = dbscan_extension / 2;
            end

            dbscan_epsilon = quantile(dbscan_kD_sorted,dbscan_midpoint);
            sprintf("Optimal number of clusters found: %d, with quantile %f", max(nb_labels), dbscan_midpoint)
        else
            dbscan_epsilon = quantile(dbscan_kD_sorted,Opts.dbscanQuantile);
        end

        % Finalize with an actual scan
        dbscan_labels = dbscan([X,logL],dbscan_epsilon,dbscan_minpts);

        unique_labels = sort(unique(dbscan_labels));
        unique_labels = unique_labels(2:end); % Skip outliers
        nb_labels = length(unique_labels);

        sprintf("Clusters found: %d", nb_labels)

        % Avoid high-likelihood outliers (misclassified)
        logL_high = quantile(logL, 0.8);

        out_mask = (dbscan_labels == -1) & (logL > logL_high);
        sprintf("Number of misclassified outliers: %d", sum(out_mask))

        % Put outliers in the most appropriate cluster (minimize distance)
        outliers = X(out_mask,:);
        outmindist = zeros(size(outliers,1), nb_labels);

        for label_index = 1:nb_labels
            pD = pdist2(outliers, X(dbscan_labels == unique_labels(label_index),:), 'euclidean');

            outmindist(:, label_index) = min(pD, [], 2);
        end

        [~,outmindist_index] = min(outmindist, [], 2); 
        
        dbscan_labels(out_mask) = unique_labels(outmindist_index);

        % Collect best options
        XP_star = cell(nb_labels, 1);
        X_star = cell(nb_labels, 1);
        pcks = cell(nb_labels,1);
        lsf = cell(nb_labels,1);
        cost_lsf = cell(nb_labels,1);

        for label_index = 1:nb_labels

            % Restrict experimental design to cluster
            label = unique_labels(label_index);
            X_label = X(dbscan_labels == label,:);
            logL_label = logL(dbscan_labels == label);

            % Clean logL outliers if enough data
            % TODO: not enough, check size
            if sum(logL_label > quantile(logL_label,0.3)) > 5
                X_label = X_label(logL_label > quantile(logL_label,0.3),:);
                logL_label = logL_label(logL_label > quantile(logL_label,0.3),:);
            end
            
            % Construct a PC-Kriging surrogate of the log-likelihood
            if isfield(Opts, 'PCK')
                PCKOpts = Opts.PCK;
            end

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
            cost_LSF = abs(mean_post_LSF) ./ (sqrt(var_post_LSF).^1.0);
            [~, opt_index] = mink(cost_LSF, Opts.lsfEvaluations);
            xp_opt = BusAnalysis.Results.Bus.PostSamples(opt_index, :);
            
            XP_star{label_index} = xp_opt;
            X_star{label_index} = BusAnalysis.Results.PostSamples(opt_index, :);
            lsf{label_index} = BusAnalysis.Results.Bus.LSF;
            cost_lsf{label_index} = cost_LSF(opt_index);
        end

        sprintf("All clustering steps done, applying merging strategy")

        % Merge optimal points
        dbscan_weights = zeros(nb_labels,1);
        U_cost = zeros(nb_labels, Opts.lsfEvaluations);
        
        for label_index =1:nb_labels

            % compute strata probabilities
            %dbscan_weights(label_index) = sum(dbscan_labels == unique_labels(label_index)) * 1.0 / size(X,1);  
            
            % Weighted by likelihood value
            logL_label = logL(dbscan_labels == unique_labels(label_index)) + Opts.Bus.logC;
            dbscan_weights(label_index) = sum(exp(logL_label));
        end

        % Take most misclassified
        %dbscan_weights = 

        % normalize weights
        dbscan_weights = dbscan_weights ./ sum(dbscan_weights);

        for j = 1:nb_labels
            
            %cost_LSF = zeros(Opts.lsfEvaluations, nb_labels);
                
            % Eval Opts.lsfEvaluations samples
            %for l = 1:nb_labels
            %    [mean_post_LSF, var_post_LSF] = uq_evalModel(lsf{l}, XP_star{j});
            %    cost_LSF(:,l) = abs(mean_post_LSF) ./ sqrt(var_post_LSF);
            %end
            
            % Eval weighted cost
            %U_cost(j,:) = transpose(normcdf(-cost_LSF) * dbscan_weights);

            % Eval uncorrelated cost
            U_cost(j,:) = dbscan_weights(j) ./ cost_lsf{j};
        end

        % Minimize column-wise, find rows where minimum lie
        [U_cost_mins, xopt_label_index] = max(U_cost);

        % Minimize row-wise, find column of minimum
        [~, xopt_sample_index] = max(U_cost_mins);

        % Get optimal point, removing the 
        xopt = X_star{xopt_label_index(xopt_sample_index)}(xopt_sample_index,:);

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

            prior_eval = zeros(size(Opts.Validation.PostSamples,1),nb_labels);
            post_eval = zeros(size(Opts.Validation.PriorSamples,1),nb_labels);

            for label_index = 1:nb_labels
                logL_PCK = pcks{opt_label_index};
                post_eval(:,label_index) = uq_evalModel(logL_PCK, Opts.Validation.PostSamples);
                prior_eval(:, label_index) = uq_evalModel(logL_PCK, Opts.Validation.PriorSamples);
            end

            sprintf("PCK LOO error: %g", logL_PCK.Error.LOO)
            sprintf("PCK Validation error: %g", logL_PCK.Error.Val)

            set(post_valid_plot, 'XData', Opts.Validation.PostLogLikelihood, 'YData', post_eval * dbscan_weights);
            set(prior_valid_plot, 'XData', Opts.Validation.PriorLogLikelihood, 'YData', prior_eval * dbscan_weights);
            set(logLhist, 'Data', logL);

            W = pca(X);
            T = X * W(:,1:2);

            in_logL_mask = logL > quantile(logL,0.2);

            set(pca_scatter, 'XData',T(in_logL_mask,1), 'YData', T(in_logL_mask,2), "CData", logL(in_logL_mask))
            set(pca_colorbar, 'Limits', [min(logL(in_logL_mask)), max(logL(in_logL_mask))])
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