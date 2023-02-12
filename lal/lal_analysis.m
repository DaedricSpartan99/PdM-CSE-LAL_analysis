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
        W = pca(X);
        T = X * W(:,1:2);
        Tpost = Opts.Validation.PostSamples * W(:,1:2);
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
            hist_plots{k}.SuSMedian = xline(mean(Opts.Validation.PostSamples(:,k)),  '--b', 'LineWidth', 5);
            hold off
            legend('Prior', 'Posterior', 'SuS-Samples', 'Min cost point', 'SuS Median')
            title(sprintf('Component %d',k))
        end
        
        drawnow
    end

    if isfield(Opts, 'StoreBusResults') && Opts.StoreBusResults
        LALAnalysis.lsfEvaluations = cell(Opts.MaximumEvaluations,1);
    end

    if ~isfield(Opts, 'Ridge')
        Opts.Ridge = 0.;
    end

    if ~isfield(Opts, 'MinCostSamples')
        Opts.MinCostSamples = 3;
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

        sprintf("Iteration number: %d", i)
        sprintf("PCK LOO error: %g", logL_PCK.Error.LOO)

        if isfield(Opts, 'Validation')
            sprintf("PCK Validation error: %g", logL_PCK.Error.Val)
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
                        x_prior = uq_getSample(Opts.Prior, 1000);

                        qxb = quantile(x_prior, 0.05);
                        qxu = quantile(x_prior, 0.95);
                        x_prior = x_prior(all(x_prior > qxb & x_prior < qxu, 2), :);

                        % Optimize from each point
                        x0 = [maxl_x0; x_prior(1:10,:)];
                        xmin = qxb;
                        xmax = qxu;
    
                        % determine c from experimental design
                        c_zero_variance = -maxl_logL;
    
                        % define inverse log-likelihood to find the minimum of
                        f = @(x) -uq_evalModel(logL_PCK, x);

                        opt_pck = zeros(size(x0,1),1);

                        for opt_ind = 1:size(x0,1)

                            % maximize surrogate log-likelihood
                            [~, maxl_pck, found_flag] = fmincon(f, x0(opt_ind,:), [], [], [], [], xmin, xmax);
        
                            % Take negative log-likelihood (overestimation)
                            if found_flag == 1
                                opt_pck(opt_ind) = -maxl_pck;
                            else
                                sprintf('Found patological value of log(c) estimation, correcting with boundary point.')
                                opt_pck(opt_ind) = -c_zero_variance;
                            end
                        end
    
                        BayesOpts.Bus.logC = min(c_zero_variance, -max(opt_pck));
    
                    case 'delaunay'
    
                        if isfield(Opts.Bus, 'Delaunay') && ~isfield(Opts.Bus.Delaunay, 'maxk')
                            Opts.Bus.Delaunay.maxk = 10;
                        end
    
                        % Rescale experimental design
                        %stdX = (X - mean(X)) ./ std(X);
                        %stdX = X ./ max(X);
                        T = delaunayn(X_cleaned, {'QbB'});
        
                        % compute midpoints and maximize variances
                        W = reshape(X_cleaned(T,:), size(T,1), size(T,2), []);
                        Wm = mean(W, 2);
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
    
                    case 'maxhull'
    
                        k = convhulln(X_cleaned, {'QbB'});
                        W = X_cleaned(k,:);
    
                        BayesOpts.Bus.logC = -max(uq_evalModel(logL_PCK , W));
    
                    case 'overmax'
    
                        % proportion
                        prop = 0.5*(1. - 2*atan(i - Opts.MaximumEvaluations/2)/pi);
        
                        % a little bit higher than peaks
                        BayesOpts.Bus.logC = -max(logL * (1. + 0.5 * prop));
                end
            end

        % Control the number of subsets, if too much
        repeat_SuS = true;
        max_SuS_repeat = 10;

        while repeat_SuS && max_SuS_repeat > 0

            repeat_SuS = false;

            sprintf("Taking constant logC: %g", BayesOpts.Bus.logC)

            BusAnalysis = bus_analysis(BayesOpts);

            % evaluate U-function on the limit state function
            % Idea: maximize misclassification probability
            px_samples = BusAnalysis.Results.Bus.PostSamples;
    
            % Filter out outliers
            qXb = quantile(px_samples, 0.1);
            qXt = quantile(px_samples, 0.9);
    
            px_samples = px_samples(all(px_samples > qXb & px_samples < qXt,2), :);
            
            % Take evaluations
            [mean_post_LSF, var_post_LSF] = uq_evalModel(BusAnalysis.Results.Bus.LSF, px_samples);
    
            if size(px_samples,1) < 10
                % Pathological points case
                fprintf("No point was found to be reliable. Repeating subset simulation with higher log(c)...\n")
                repeat_SuS = true;
            end
 
            % Check for enough posterior samples
            if sum(mean_post_LSF < 0) < 0.05 * size(mean_post_LSF,1)
                fprintf("There aren't enough posterior samples. Repeating subset simulation with higher log(c)...\n")
                fprintf("Proportion: %f\n", 1.0 * sum(mean_post_LSF < 0) / size(mean_post_LSF,1))
                repeat_SuS = true;
            end    
            
            if repeat_SuS
                BayesOpts.Bus.logC = mean([BayesOpts.Bus.logC, -max(logL)]);
            end

            max_SuS_repeat = max_SuS_repeat - 1;
        end

        % Take only posterior samples
        %qmean = 10.;
        
        %px_samples = px_samples(mean_post_LSF < qmean,:);
        %mean_post_LSF = mean_post_LSF(mean_post_LSF < qmean);
        %var_post_LSF = var_post_LSF(mean_post_LSF < qmean);
        
        x_medians = median(px_samples(:,2:end)); % TODO: distance from experimental design
        
        ridge_cost = sum(abs(px_samples(:,2:end) ./ x_medians - 1),2);
        cost_LSF = abs(mean_post_LSF) ./ sqrt(var_post_LSF);
        
        % Take first three candidates
        [~, opt_index] = mink(cost_LSF + Opts.Ridge * ridge_cost * median(cost_LSF), Opts.MinCostSamples);
        xopt = px_samples(opt_index, 2:end);
        
        % Choose the one maximizing log-likelihood
        [~, opt_index] = max(uq_evalModel(logL_PCK, xopt));
        xopt = xopt(opt_index,:);
    
        fprintf("Optimal X chosen to: ")
        display(xopt)
        fprintf("\n")
        
        % Add to experimental design
        X = [X; xopt];
        logL = [logL; Opts.LogLikelihood(xopt) ];

        % Store result as history
        LALAnalysis.OptPoints(i).X = xopt;
        LALAnalysis.OptPoints(i).logL = logL(end);
        LALAnalysis.OptPoints(i).lsf = cost_LSF(opt_index);
        LALAnalysis.PCK(i) = logL_PCK;

        if isfield(Opts, 'StoreBusResults') && Opts.StoreBusResults
            % Store in results
            LALAnalysis.BusAnalysis(i) = BusAnalysis;
            % Store evaluations
            LALAnalysis.lsfEvaluations{i} = cost_LSF;
            % Store target
            LALAnalysis.logC(i) = BayesOpts.Bus.logC;
        end

        %if abs(logL(end) - min(logL)) < eps
        %    sprintf("Found unconvenient point, logL = %g", logL(end))
        %end

        % Update plot
        if Opts.PlotLogLikelihood

            set(post_valid_plot, 'XData', Opts.Validation.PostLogLikelihood, 'YData', uq_evalModel(logL_PCK, Opts.Validation.PostSamples));
            set(prior_valid_plot, 'XData', Opts.Validation.PriorLogLikelihood, 'YData', uq_evalModel(logL_PCK, Opts.Validation.PriorSamples));
            histogram(ax3, logL_cleaned, 12);
            %set(logLhist, 'Data', logL_cleaned, 'BinLimits', [min(logL_cleaned), max(logL_cleaned)]);

            W = pca(X_cleaned);
            T = X_cleaned * W(:,1:2);
            Tpost = Opts.Validation.PostSamples * W(:,1:2);
            set(pca_scatter, 'XData',T(:,1), 'YData', T(:,2), "CData", logL_cleaned)
            set(pca_post_scatter, 'XData', Tpost(:,1), 'YData', Tpost(:,2))
            set(pca_colorbar, 'Limits', [min(logL_cleaned), max(logL_cleaned)])
            %pca_scatter = scatter(T(:,1), T(:,2), "ColorVariable", logL, 'Filled')

            % Histogram plots
            for k = 1:size(X,2)
                set(hist_plots{k}.SuS, 'Data', px_samples(:,k+1))
                set(hist_plots{k}.Opt, 'Value', xopt(k));
                set(hist_plots{k}.SuSMedian, 'Value', x_medians(k));
            end

            drawnow
        end
    end

    % Store results
    LALAnalysis.ExpDesign.X = X;
    LALAnalysis.ExpDesign.LogLikelihood = logL;
    LALAnalysis.Opts = Opts;
end