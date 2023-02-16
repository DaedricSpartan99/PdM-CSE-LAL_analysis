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

    % Handles
    log_prior = @(x) uq_evalLogPDF(x, Opts.Prior);

    % Initialize output following initial guesses
    if isfield(Opts.ExpDesign, 'InitEval')

        X = uq_getSample(Opts.Prior, Opts.ExpDesign.InitEval); % 'LHS'

        logL = Opts.LogLikelihood(X);
    else
        X = Opts.ExpDesign.X;
        logL = Opts.ExpDesign.LogLikelihood;
    end

    if ~isfield(Opts, 'PlotLogLikelihood')
        Opts.PlotLogLikelihood = false;
    end

    post = logL + log_prior(X);

    %Opts.Validation.PostLogLikelihood = max(Opts.Validation.PostLogLikelihood, -1200);
    %Opts.Validation.PriorLogLikelihood = max(Opts.Validation.PriorLogLikelihood, -1200);

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
        histogram(ax3, logL, 12);
        title(ax3, 'Experimental design emplacement')
        xlabel('Log-likelihood')

        ax4 = nexttile;
        W_pca = pca(X);
        T = X * W_pca(:,1:2);
        Tpost = Opts.Validation.PostSamples * W_pca(:,1:2);
        hold on
        pca_scatter = scatter(ax4, T(:,1), T(:,2), 20, logL, 'Filled')
        pca_colorbar = colorbar(ax4)
        pca_post_scatter = scatter(ax4, Tpost(:,1), Tpost(:,2), 5)
        hold off
        title(ax4, 'Experimental design PCA')
        xlabel(ax4, 'x1')
        ylabel(ax4, 'x2')

        % Histogram figure
        figure
        m_tile = factor(size(X,2));
        m_tile = m_tile(1);
        tiledlayout(size(X,2)/m_tile,m_tile)

        hist_plots = cell(size(X,2),1);

        for k = 1:size(X,2)
            hist_plots{k}.ax = nexttile

            hold on
            hist_plots{k}.Prior = histogram(Opts.Validation.PriorSamples(:,k),50);
            hist_plots{k}.Post = histogram(Opts.Validation.PostSamples(:,k),50); 
            hist_plots{k}.SuS = histogram(Opts.Validation.PostSamples(:,k),50); 
            hist_plots{k}.Opt = xline(mean(Opts.Validation.PostSamples(:,k)), 'LineWidth', 5);
            %hist_plots{k}.SuSMedian = xline(mean(Opts.Validation.PostSamples(:,k)),  '--b', 'LineWidth', 5);
            hold off
            legend('Prior', 'Posterior', 'SuS-Samples', 'Min cost point')
            title(sprintf('Component %d',k))
        end
        
        drawnow
    end

    if isfield(Opts, 'StoreBusResults') && Opts.StoreBusResults
        LALAnalysis.lsfEvaluations = cell(Opts.MaximumEvaluations,1);
    end

    if ~isfield(Opts, 'SelectMax')
        Opts.SelectMax = 1;
    end

    if ~isfield(Opts, 'ClusterRange')
        Opts.ClusterRange = 2:6;
    end

    if ~isfield(Opts, 'ClusterMaxIter')
        Opts.ClusterMaxIter = 50;
    end

    %post_input = Opts.Prior;
    
    % Begin iterations
    for i = 1:Opts.MaximumEvaluations

        % Construct a PC-Kriging surrogate of the log-likelihood
        if isfield(Opts, 'PCK')
            PCKOpts = Opts.PCK;
        end

        % Create definitive PCK
        PCKOpts.Type = 'Metamodel';
        PCKOpts.MetaType = 'PCK';
        PCKOpts.Mode = 'optimal';   
        %PCKOpts.FullModel = Opts.LogLikelihood;
        PCKOpts.Input = Opts.Prior; 
        PCKOpts.isVectorized = true;
        
        if isfield(Opts, 'Validation')
            PCKOpts.ValidationSet.X = Opts.Validation.PostSamples;
            PCKOpts.ValidationSet.Y = Opts.Validation.PostLogLikelihood;
        end

        % Address instabilities in the experimental design (0.05 quantile)
        if isfield(Opts, 'cleanQuantile')   
            in_logL_mask = logL > quantile(logL,Opts.cleanQuantile);
            X_cleaned = X(in_logL_mask,:);
            logL_cleaned = logL(in_logL_mask); 
        else

            X_cleaned = X;
            logL_cleaned = logL;
        end

        % Create definitive PCK
        PCKOpts.ExpDesign.X = X_cleaned;
        PCKOpts.ExpDesign.Y = logL_cleaned;

        logL_PCK = uq_createModel(PCKOpts, '-private');

        fprintf("Iteration number: %d\n", i)
        fprintf("PCK LOO error: %g\n", logL_PCK.Error.LOO)

        if isfield(Opts, 'Validation')
            fprintf("PCK Validation error: %g\n", logL_PCK.Error.Val)
        end
        
        % TODO: Determine optimal c = 1 / max(L)

        % Execute Bayesian Analysis in Bus framework
        BayesOpts.Prior = Opts.Prior;

        if isfield(Opts, 'Bus')
            BayesOpts.Bus = Opts.Bus;
        else
            BayesOpts.Bus.CStrategy = 'max';
        end

        BayesOpts.LogLikelihood = logL_PCK; 

        % Adaptively determine constant Bus.logC
            % TODO: better algorithm
            if ~isfield(Opts.Bus, 'logC')
    
                % Default strategy
                if ~isfield(BayesOpts.Bus, 'CStrategy')
                    BayesOpts.Bus.CStrategy = 'max';
                end
               
                % Take specified strategy
                switch BayesOpts.Bus.CStrategy
                    case 'max'
                        BayesOpts.Bus.logC = -max(logL);
                    case 'latest'
                        BayesOpts.Bus.logC = -logL(end);
                    case 'maxpck' 
    
                        % get maximum of experimental design
                        [maxl_logL, maxl_index] = max(logL_cleaned);
                        maxl_x0 = X_cleaned(maxl_index, :);

                        % sample from prior distribution for other points
                        x_prior = uq_getSample(Opts.Prior, 5000);

                        qxb = quantile(x_prior, 0.025);
                        qxu = quantile(x_prior, 0.975);
                        x_prior = x_prior(all(x_prior > qxb & x_prior < qxu, 2), :);

                        % Optimize from each point
                        x0 = [maxl_x0; x_prior(1:10,:)];
                        xmin = min(x_prior);
                        xmax = max(x_prior);
    
                        % determine c from experimental design
                        c_zero_variance = -maxl_logL;
    
                        % define inverse log-likelihood to find the minimum of
                        f = @(x) -uq_evalModel(logL_PCK, x);

                        opt_pck = zeros(size(x0,1),1);

                        for opt_ind = 1:size(x0,1)

                            % maximize surrogate log-likelihood
                            options = optimoptions('fmincon', 'Display', 'off');
                            [~, maxl_pck, found_flag] = fmincon(f, x0(opt_ind,:), [], [], [], [], xmin, xmax, [], options);
        
                            % Take negative log-likelihood (overestimation)
                            if found_flag >= 0
                                opt_pck(opt_ind) = -maxl_pck;
                            else
                                fprintf('Found patological value of log(c) estimation, correcting with experimental design maximum.\n')
                                opt_pck(opt_ind) = -c_zero_variance;
                            end
                            fprintf("Peak index %d, fmincon flag: %d, log-likelihood: %f \n", opt_ind, found_flag, -maxl_pck)
                        end
    
                        BayesOpts.Bus.logC = min(c_zero_variance, -max(opt_pck));
                        %BayesOpts.Bus.logC = -max(opt_pck);
    
                    case 'delaunay'
    
                        if isfield(Opts.Bus, 'Delaunay') && ~isfield(Opts.Bus.Delaunay, 'maxk')
                            Opts.Bus.Delaunay.maxk = 10;
                        end
    
                        % Rescale experimental design
                        %stdX = (X - mean(X)) ./ std(X);
                        %stdX = X ./ max(X);
                        T = delaunayn(X_cleaned, {'QbB'});
        
                        % compute midpoints and maximize variances
                        W_del = reshape(X_cleaned(T,:), size(T,1), size(T,2), []);
                        Wm = mean(W_del, 2);
                        midpoints = permute(Wm, [1,3,2]);
                        [mmeans, mvars] = uq_evalModel(logL_PCK , midpoints);
                        
                        % get only a certain number of max variance
                        [~, varindex] = maxk(mvars, Opts.Bus.Delaunay.maxk);
                        midpoints = midpoints(varindex,:);
                        mmeans = mmeans(varindex);
    
                        % sort by greatest mean
                        [~, meanindex] = sort(mmeans, 'descend');
                        midpoints = midpoints(meanindex,:);
        
                        BayesOpts.Bus.logC = -uq_evalModel(logL_PCK , midpoints(1,:));
                end
            end

        %% DEPRECATED

        % Control the number of subsets, if too much
        repeat_SuS = true; % TODO: check with clustering
        max_SuS_repeat = 50;

        while repeat_SuS && max_SuS_repeat > 0

            repeat_SuS = false;

            fprintf("Taking constant logC: %g\n", BayesOpts.Bus.logC)

            BusAnalysis = bus_analysis(BayesOpts);

            % evaluate U-function on the limit state function
            % Idea: maximize misclassification probability
            px_samples = BusAnalysis.Results.Bus.PostSamples;
    
            % Filter out outliers
            %qXb = quantile(px_samples, 0.025);
            %qXt = quantile(px_samples, 0.975);
    
            %px_samples = px_samples(all(px_samples > qXb & px_samples < qXt,2), :);
            
            % Take evaluations
            [mean_post_LSF, ~] = uq_evalModel(BusAnalysis.Results.Bus.LSF, px_samples);
    
            if size(px_samples,1) < 10
                % Pathological points case
                fprintf("No point was found to be reliable. Repeating subset simulation with higher log(c)...\n")
                repeat_SuS = true;
            end
 
            % Check for enough posterior samples
            sus_ratio = 1.0 * sum(mean_post_LSF < 0) / size(mean_post_LSF,1);
            if sus_ratio < 0.1
                fprintf("There aren't enough posterior samples. Repeating subset simulation with higher log(c)...\n")
                fprintf("Proportion: %f\n", sus_ratio)
                repeat_SuS = true;
            end    
            
            if repeat_SuS

                %h = BusAnalysis.Results.Subset.History.q;

                %if length(h) < 2 || sum(mean_post_LSF < 0) == 0
                    BayesOpts.Bus.logC = mean([BayesOpts.Bus.logC, -max(logL)]);
                %else
                    % Take last subset threshold
                    %BayesOpts.Bus.logC = BayesOpts.Bus.logC + quantile(mean_post_LSF,0.1);
                %end
            end

            fprintf("Taking constant logC: %g\n", BayesOpts.Bus.logC)

            max_SuS_repeat = max_SuS_repeat - 1;
        end

        %%

        %fprintf("Taking constant logC: %g\n", BayesOpts.Bus.logC)
        %BusAnalysis = bus_analysis(BayesOpts);

        % evaluate U-function on the limit state function
        % Idea: maximize misclassification probability
        %px_samples = BusAnalysis.Results.Bus.PostSamples;

        % squeez SuS samples
        %x_samples = px_samples(:,2:end);
        %x_samples = x_samples ./ max(x_samples);

        %minpts = 50;
        %kD = pdist2(x_samples,x_samples,'euc','Smallest',minpts);
        %kD = sort(kD(end,:));
        %%[~,eps_dbscan_ind] = knee_pt(kD, 1:length(kD));
        %eps_dbscan = kD(eps_dbscan_ind);

        %dbscan_labels = dbscan(x_samples, eps_dbscan,minpts);

        %px_samples = px_samples(dbscan_labels ~= -1, :);

        % Filter out outliers
        %qXb = quantile(px_samples(:,2:end), 0.025);
        %qXt = quantile(px_samples(:,2:end), 0.975);
        %px_samples = px_samples(all(px_samples(:,2:end) > qXb & px_samples(:,2:end) < qXt,2), :);            
      
        % Take lsf evaluations
        [mean_post_LSF, var_post_LSF] = uq_evalModel(BusAnalysis.Results.Bus.LSF, px_samples);
    
        % Compute surrogate log-likelihood
        %logL_pck_samples = uq_evalModel(logL_PCK, px_samples(:,2:end));

        % Compute U-function and misclassification probability
        cost_LSF = abs(mean_post_LSF) ./ sqrt(var_post_LSF);
        W = normcdf(-cost_LSF);
        
        % Normalize data before clustering
        x_mean = mean(px_samples(:,2:end));
        x_std = std(px_samples(:,2:end));
        x_norm = (px_samples(:,2:end) - x_mean) ./ x_std;

        % Cluster (TODO: adapt estimating the number of peaks);
        if length(Opts.ClusterRange) == 1
            [cost_labels, xopt] = kw_means(x_norm, W, max(Opts.ClusterRange), Opts.ClusterMaxIter); 
        else
            [cost_labels, xopt] = w_means(x_norm, W, Opts.ClusterRange, Opts.ClusterMaxIter);
        end

        % Un-normalize xopt and sort by weights
        c_weights = zeros(size(xopt,1),1);
        for j = 1:length(c_weights)
            p_j = px_samples(cost_labels == j,1);
            w_j = W(cost_labels == j);

            %xopt(j,:) = colwise_weightedMedian(px_samples(cost_labels == j,2:end),w_j);
            xopt(j,:) = sum(px_samples(cost_labels == j,2:end) .* w_j) / sum(w_j);

            % Take greatest likelihood points
            c_weights(j) = mean(p_j .* w_j);
            %c_weights(j) = weightedMedian(w_j, p_j);
        end

        [c_weights, opt_ind] = maxk(c_weights, min(Opts.SelectMax, length(c_weights)));
        xopt = xopt(opt_ind,:);       
    
        fprintf("Optimal X chosen to: ")
        display(xopt)
        fprintf("\n")

        fprintf("With cluster cost: ")
        display(c_weights)
        fprintf("\n")

        % Compute surrogate log-likelihood
        logL_pck_opt = uq_evalModel(logL_PCK, xopt);

        fprintf("Optimal points surrogate log-likelihood: ")
        display(logL_pck_opt)
        fprintf("\n")

        % Compute real log-likelihood
        logL_opt = Opts.LogLikelihood(xopt);
        
        fprintf("Optimal points real log-likelihood: ")
        display(logL_opt)
        fprintf("\n")
        
        % Add to experimental design
        X = [X; xopt];
        logL = [logL; logL_opt ];
        post = [post; logL_opt + log_prior(xopt)];

        % Store result as history
        LALAnalysis.OptPoints(i).X = xopt;
        LALAnalysis.OptPoints(i).logL = logL(end);
        %LALAnalysis.OptPoints(i).lsf = uq_evalModel(BusAnalysis.Results.Bus.LSF, centroids);
        LALAnalysis.PCK(i) = logL_PCK;

        if isfield(Opts, 'StoreBusResults') && Opts.StoreBusResults
            % Store in results
            LALAnalysis.BusAnalysis(i) = BusAnalysis;
            % Store evaluations
            LALAnalysis.lsfEvaluations{i} = cost_LSF;
            % Store target
            LALAnalysis.logC(i) = BayesOpts.Bus.logC;
        end


        % Update plot
        if Opts.PlotLogLikelihood

            set(post_valid_plot, 'XData', Opts.Validation.PostLogLikelihood, 'YData', uq_evalModel(logL_PCK, Opts.Validation.PostSamples));
            set(prior_valid_plot, 'XData', Opts.Validation.PriorLogLikelihood, 'YData', uq_evalModel(logL_PCK, Opts.Validation.PriorSamples));
            histogram(ax3, logL_cleaned, 12);
            %set(logLhist, 'Data', logL_cleaned, 'BinLimits', [min(logL_cleaned), max(logL_cleaned)]);

            W_pca = pca(X_cleaned);
            T = X_cleaned * W_pca(:,1:2);
            Tpost = Opts.Validation.PostSamples * W_pca(:,1:2);
            set(pca_scatter, 'XData',T(:,1), 'YData', T(:,2), "CData", logL_cleaned)
            set(pca_post_scatter, 'XData', Tpost(:,1), 'YData', Tpost(:,2))
            set(pca_colorbar, 'Limits', [min(logL_cleaned), max(logL_cleaned)])
            %pca_scatter = scatter(T(:,1), T(:,2), "ColorVariable", logL, 'Filled')

            % Histogram plots
            for k = 1:size(X,2)
                set(hist_plots{k}.SuS, 'Data', px_samples(:,k+1))
                set(hist_plots{k}.Opt, 'Value', xopt(1,k));
                %set(hist_plots{k}.SuSMedian, 'Value', x_medians(k));
            end

            drawnow
        end
    end

    % Store results
    LALAnalysis.ExpDesign.X = X;
    LALAnalysis.ExpDesign.LogLikelihood = logL;
    LALAnalysis.ExpDesign.UnNormPosterior = post;
    LALAnalysis.Opts = Opts;
end