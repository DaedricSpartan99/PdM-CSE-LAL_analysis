clearvars
rng(100,'twister')
uqlab

addpath('../lal')

%% visualize

%uq_figure
%[I,~] = imread('SimplySupportedBeam.png');
%image(I)
%axis equal
%set(gca, 'visible', 'off')

%% forward model setup

ModelOpts.mFile = 'uq_SimplySupportedBeam';
ModelOpts.isVectorized = true;

myForwardModel = uq_createModel(ModelOpts);

%% prior setup

PriorOpts.Marginals(1).Name = 'b';               % beam width
PriorOpts.Marginals(1).Type = 'Constant';
PriorOpts.Marginals(1).Parameters = [0.15];      % (m)

PriorOpts.Marginals(2).Name = 'h';               % beam height
PriorOpts.Marginals(2).Type = 'Constant';
PriorOpts.Marginals(2).Parameters = [0.3];       % (m)

PriorOpts.Marginals(3).Name = 'L';               % beam length
PriorOpts.Marginals(3).Type = 'Constant';
PriorOpts.Marginals(3).Parameters = 5;           % (m)

PriorOpts.Marginals(4).Name = 'E';               % Young's modulus
PriorOpts.Marginals(4).Type = 'LogNormal';
PriorOpts.Marginals(4).Moments = [30 4.5]*1e9;   % (N/m^2)

PriorOpts.Marginals(5).Name = 'p';               % uniform load
PriorOpts.Marginals(5).Type = 'Constant';
PriorOpts.Marginals(5).Parameters = 1.2317e+04;           % (N/m)
%PriorOpts.Marginals(5).Type = 'Gaussian';
%PriorOpts.Marginals(5).Moments = [12000 600]; % (N/m)

myPriorDist = uq_createInput(PriorOpts);

%% Measurement setup

myData.y = [12.84; 13.12; 12.13; 12.19; 12.67]/1000; % (m)
myData.Name = 'Mid-span deflection';

DiscrepancyOpts(1).Type = 'Gaussian';
DiscrepancyOpts(1).Parameters = var(myData.y);

%% Bayesian invertion

BayesOpts.Type = 'Inversion';
BayesOpts.Data = myData;
%BayesOpts.Discrepancy = DiscrepancyOpts;

refBayesAnalysis = uq_createAnalysis(BayesOpts);

uq_postProcessInversionMCMC(refBayesAnalysis);

%% post sample exctraction and clean up

M = size(refBayesAnalysis.Results.PostProc.PostSample,2); % number of time-steps
Solver.MCMC.NChains = refBayesAnalysis.Internal.Solver.MCMC.NChains;

post_samples = permute(refBayesAnalysis.Results.PostProc.PostSample, [1, 3, 2]);
post_samples = reshape(post_samples, [], M);
post_logL_samples = reshape(refBayesAnalysis.Results.PostProc.PostLogLikeliEval, [], 1);

post_samples = post_samples(post_logL_samples > quantile(post_logL_samples, 0.1), :);
post_logL_samples = post_logL_samples(post_logL_samples > quantile(post_logL_samples, 0.1));
post_samples_size = size(post_samples, 1); 

% prepare prior samples
prior_samples = uq_getSample(refBayesAnalysis.Internal.FullPrior, post_samples_size);
prior_logL_samples = refBayesAnalysis.LogLikelihood(prior_samples);

prior_samples = prior_samples(prior_logL_samples > quantile(prior_logL_samples, 0.1), :);
prior_logL_samples = prior_logL_samples(prior_logL_samples > quantile(prior_logL_samples, 0.1));

%% Experimental design setup

init_eval = 10;
log_likelihood = refBayesAnalysis.LogLikelihood;

LALOpts.ExpDesign.X = uq_getSample(refBayesAnalysis.Internal.FullPrior, init_eval);
LALOpts.ExpDesign.LogLikelihood = log_likelihood(LALOpts.ExpDesign.X);

init_X = LALOpts.ExpDesign.X;
init_logL = LALOpts.ExpDesign.LogLikelihood;

%% Plot of the likelihood in components a1 and a2

a1 = 1;
a2 = 2;

lq = quantile(init_logL, 0.2);
init_Xq = init_X(init_logL > lq,:);
init_logLq = init_logL(init_logL > lq);

Hplot = linspace(min(init_Xq(:,a1)), max(init_Xq(:,a1)), 50);
Lplot = linspace(min(init_Xq(:,a2)), max(init_Xq(:,a2)), 50);

[x_grid_1, x_grid_2] = meshgrid(Hplot,Lplot);
HL_grid = [x_grid_1(:), x_grid_2(:)];

%Xplot = [0.5524, 0.0309, 0.8929, 0.0279, 0, 0];
Xplot = mean(init_Xq);
Xplot = [repmat(Xplot, size(HL_grid,1),1), HL_grid];

Xplot(:,[a1,a2]) = HL_grid;

logL_grid = log_likelihood(Xplot);
logL_grid = reshape(logL_grid, 50, 50);


%% Construct a PCK which fits good

PCKOpts.Type = 'Metamodel';
PCKOpts.MetaType = 'PCK';
PCKOpts.Mode = 'optimal';  
        
PCKOpts.Input = refBayesAnalysis.Internal.FullPrior; 
PCKOpts.isVectorized = true;
PCKOpts.ExpDesign.X = init_X;
PCKOpts.ExpDesign.Y = init_logL;

PCKOpts.PCK.PCE.Degree = 1:10;

%LALOpts.PCK.PCE.PolyTypes = {'Hermite', 'Hermite'};
%LALOpts.PCK.Optim.Method = 'CMAES';
%LALOpts.PCK.Kriging.Optim.MaxIter = 1000;
%PCKOpts.PCK.Kriging.Corr.Family = 'Gaussian';
%PCKOpts.PCK.Kriging.Corr.Family = 'Matern-5_2';
%PCKOpts.PCK.Kriging.Corr.Type = 'Separable';
%PCKOpts.PCK.Kriging.Corr.Type = 'ellipsoidal';
%LALOpts.PCK.Kriging.theta = 9.999;
%LALOpts.PCK.Display = 'verbose';

PCKOpts.ValidationSet.X = prior_samples;
PCKOpts.ValidationSet.Y = prior_logL_samples;

logL_PCK = uq_createModel(PCKOpts);

fprintf("---   Leave-one-out error: %f\n", logL_PCK.Error.LOO)
fprintf("---   Validation error: %f\n", logL_PCK.Error.Val)

figure
check_interval = [min(prior_logL_samples), max(prior_logL_samples)];
prior_evals = uq_evalModel(logL_PCK, prior_samples);

hold on
plot(check_interval , check_interval);
scatter(prior_logL_samples, prior_evals);
hold off
title('Prior samples')
ylabel('Surrogate Log-Likelihood')
xlabel('Real Log-Likelihood')
xlim(check_interval)
ylim(check_interval)

drawnow

%logL_PCK_grid = uq_evalModel(logL_PCK, Xplot);
%logL_PCK_grid = reshape(logL_PCK_grid, 50, 50);



%% Bayesian analysis (tuning first peaks step)

clear LALOpts

%LALOpts.Bus.logC = 0; %-max(post_logL_samples); % best value: -max log(L) 
%LALOpts.Bus.p0 = 0.1;                            % Quantile probability for Subset
%LALOpts.Bus.BatchSize = 1e3;                             % Number of samples for Subset simulation
%LALOpts.Bus.MaxSampleSize = 1e4;
LALOpts.MaximumEvaluations = 20;
LALOpts.ExpDesign.X = init_X;
LALOpts.ExpDesign.LogLikelihood = init_logL;
LALOpts.PlotLogLikelihood = true;
LALOpts.Bus.CStrategy = 'maxpck';
%LALOpts.MinCostSamples = 10;  
LALOpts.SelectMax = 1;
LALOpts.ClusterRange = 2:15;

LALOpts.PCK.PCE.Degree = 1;
%LALOpts.PCK.PCE.PolyTypes = {'Hermite', 'Hermite'};
%LALOpts.PCK.Optim.Method = 'CMAES';
%LALOpts.PCK.Kriging.Optim.MaxIter = 1000;
%LALOpts.PCK.Kriging.Corr.Family = 'Gaussian';
%LALOpts.PCK.Kriging.Corr.Family = 'Matern-3_2';
%LALOpts.PCK.Kriging.Corr.Type = 'Separable';
%LALOpts.PCK.Kriging.Corr.Type = 'ellipsoidal';
%LALOpts.PCK.Kriging.theta = 9.999;
%LALOpts.PCK.Display = 'verbose';

%LALOpts.cleanQuantile = 0.05;

LALOpts.LogLikelihood = refBayesAnalysis.LogLikelihood;
LALOpts.Prior = refBayesAnalysis.Internal.FullPrior;

LALOpts.Validation.PostSamples = post_samples;
LALOpts.Validation.PostLogLikelihood = post_logL_samples;
LALOpts.Validation.PriorSamples = prior_samples;
LALOpts.Validation.PriorLogLikelihood = prior_logL_samples;

LALOpts.StoreBusResults = true;

FirstLALAnalysis = lal_analysis(LALOpts);

fprintf("---   LogC: %f\n", FirstLALAnalysis.logC(end));
fprintf("---   Found point with likelihood: %f\n", FirstLALAnalysis.OptPoints(end).logL)


xopt = FirstLALAnalysis.OptPoints(end).X;
logL_PCK = FirstLALAnalysis.PCK(end);

fprintf("---   The surrogate likelihood was: %f\n", uq_evalModel(logL_PCK, xopt))

logL_PCK_grid = uq_evalModel(logL_PCK, Xplot);
logL_PCK_grid = reshape(logL_PCK_grid, 50, 50);


% plot figures
figure
tiledlayout(1,2)

ax = nexttile;

hold on
contourf(ax, Hplot, Lplot, logL_grid);
colorbar(ax)
scatter(ax, init_Xq(:,a1), init_Xq(:,a2), 25, init_logLq,  'filled')
%surfplot.EdgeColor = 'none';
%surfplot.FaceAlpha = 0.5;
%surfplot_pck = surf(Hplot, Lplot, logL_PCK_grid);
%surfplot_pck.EdgeColor = 'none';
%surfplot_pck.FaceAlpha = 0.8;
hold off
title('Real likelihood visualization of component 1 and 2')

ax_pck = nexttile;

hold on
contourf(ax_pck, Hplot, Lplot, logL_PCK_grid);
colorbar(ax_pck)
scatter(ax_pck, init_Xq(:,a1), init_Xq(:,a2), 25, init_logLq,  'filled')
scatter(ax_pck, xopt(:,a1), xopt(:,a2), 45, "black")
%surfplot.EdgeColor = 'none';
%surfplot.FaceAlpha = 0.5;
%surfplot_pck = surf(Hplot, Lplot, logL_PCK_grid);
%surfplot_pck.EdgeColor = 'none';
%surfplot_pck.FaceAlpha = 0.8;
hold off
title('Surrogate PCK likelihood visualization of component 1 and 2')

sgtitle('Initial state log-likelihood')

drawnow

%% Finalize

clear LALOpts

exp_design = FirstLALAnalysis.ExpDesign;

LALOpts.PCK.PCE.Degree = 1:10;
%LALOpts.PCK.Kriging.Corr.Type = 'Separable';

LALOpts.Bus.BatchSize = 5000;
LALOpts.Bus.MaxSampleSize = 500000;

LALOpts.LogLikelihood = refBayesAnalysis.LogLikelihood;
LALOpts.Prior = refBayesAnalysis.Internal.FullPrior;

%LALOpts.MinCostSamples = 10;
%LALOpts.cleanQuantile = 0.15;
%LALOpts.Ridge = 0.;
LALOpts.SelectMax = 1;
LALOpts.ClusterRange = 2:15;

LALOpts.Validation.PostSamples = post_samples;
LALOpts.Validation.PostLogLikelihood = post_logL_samples;
LALOpts.Validation.PriorSamples = prior_samples;
LALOpts.Validation.PriorLogLikelihood = prior_logL_samples;

LALOpts.MaximumEvaluations = 20;
LALOpts.ExpDesign = exp_design;
LALOpts.Bus.CStrategy = 'maxpck';
%LALOpts.Bus.Delaunay.maxk = 15;
LALOpts.PlotLogLikelihood = true;

LALOpts.StoreBusResults = true;

LALAnalysis = lal_analysis(LALOpts);

%% Analyisis

X = LALAnalysis.ExpDesign.X;
TT = [X; post_samples];
idx = [ones(size(X,1),1); zeros(post_samples_size,1)];

figure
gplotmatrix(TT,TT,idx)

%% Plot of prior validation errors over last run

iterations = LALOpts.MaximumEvaluations;

pck_post_err = zeros(1,iterations);
pck_loo_err = zeros(1,iterations);
pck_prior_err = zeros(1,iterations);

for i = 1:iterations
    dl_prior = abs(uq_evalModel(LALAnalysis.BusAnalysis(i).Opts.LogLikelihood, prior_samples) - prior_logL_samples);
    dl_post = abs(uq_evalModel(LALAnalysis.BusAnalysis(i).Opts.LogLikelihood, post_samples) - post_logL_samples);

    pck_prior_err(i) = mean(dl_prior.^2);
    %pck_post_err(i) = mean(exp(prior_logL_samples) .* dl.^2);
    pck_post_err(i) = mean(dl_post.^2);
    pck_loo_err(i) = LALAnalysis.BusAnalysis(i).Opts.LogLikelihood.Error.LOO;
end

% Fit errors
[a,~] = polyfit(1:iterations, log10(pck_post_err), 1);
[b,~] = polyfit(iterations/2:iterations, log10(pck_loo_err(end/2:end)), 1);
[c,~] = polyfit(1:iterations, log10(pck_prior_err), 1);

figure
hold on
plot(1:iterations, 10.^(c(1) .* (1:iterations) + c(2)), 'DisplayName', 'Prior error Fit')
plot(1:iterations, 10.^(a(1) .* (1:iterations) + a(2)), 'DisplayName', 'Post error Fit')
plot(1:iterations, 10.^(b(1) .* (1:iterations) + b(2)), 'DisplayName', 'LOO error Fit')
scatter(1:iterations, pck_prior_err, 'filled', 'DisplayName', 'PCK prior validation error')
scatter(1:iterations, pck_post_err, 'filled', 'DisplayName', 'PCK posterior validation error')
scatter(1:iterations, pck_loo_err, 'filled', 'DisplayName', 'PCK LOO error')
hold off
set(gca, 'YScale', 'log')
xlabel('Iteration')
ylabel('Error')
title('Log-Likelihood PCK error plots')
legend('interpreter','latex')

sprintf("Prior validation convergence rate: %f", -c(1))
sprintf("Posterior validation convergence rate: %f", -a(1))
sprintf("LOO convergence rate: %f", -b(1))

%% Plot of the likelihood in components a1 and a2 (final state)

a1 = 1;
a2 = 2;

lq = quantile(init_logL, 0.2);
Xq = LALAnalysis.ExpDesign.X(LALAnalysis.ExpDesign.LogLikelihood > lq,:);
logLq = LALAnalysis.ExpDesign.LogLikelihood(LALAnalysis.ExpDesign.LogLikelihood > lq);

Hplot = linspace(min(Xq(:,a1)), max(Xq(:,a1)), 50);
Lplot = linspace(min(Xq(:,a2)), max(Xq(:,a2)), 50);

[x_grid_1, x_grid_2] = meshgrid(Hplot,Lplot);
HL_grid = [x_grid_1(:), x_grid_2(:)];

%Xplot = [0.5524, 0.0309, 0.8929, 0.0279, 0, 0];
%Xplot = mean(init_Xq);
Xplot = mean(post_samples);
Xplot = [repmat(Xplot, size(HL_grid,1),1), HL_grid];

Xplot(:,[a1,a2]) = HL_grid;

logL_grid = log_likelihood(Xplot);
logL_grid = reshape(logL_grid, 50, 50);

fprintf("---   LogC: %f\n", FirstLALAnalysis.logC(end));
fprintf("---   Found point with likelihood: %f\n", FirstLALAnalysis.OptPoints(end).logL)


xopt = LALAnalysis.OptPoints(end).X;
logL_PCK = LALAnalysis.PCK(end);

fprintf("---   The surrogate likelihood was: %f\n", uq_evalModel(logL_PCK, xopt))

logL_PCK_grid = uq_evalModel(logL_PCK, Xplot);
logL_PCK_grid = reshape(logL_PCK_grid, 50, 50);

% plot figures
figure
tiledlayout(1,2)

ax = nexttile;

hold on
contourf(ax, Hplot, Lplot, logL_grid);
colorbar(ax)
scatter(ax, Xq(:,a1), Xq(:,a2), 25, logLq,  'filled')
hold off
title('Real likelihood visualization of component 1 and 2')

ax_pck = nexttile;

hold on
contourf(ax_pck, Hplot, Lplot, logL_PCK_grid);
colorbar(ax_pck)
scatter(ax_pck, Xq(:,a1), Xq(:,a2), 25, logLq,  'filled')
scatter(ax_pck, xopt(:,a1), xopt(:,a2), 45, "black")
hold off
title('Surrogate PCK likelihood visualization of component 1 and 3')

sgtitle('Final state log-likelihood')
drawnow

%% Limit state function check

figure
tiledlayout(size(prior_samples,2),1)

for k = 1:size(prior_samples,2)
    nexttile

    hold on
    histogram(prior_samples(:,k),50);
    histogram(LALAnalysis.BusAnalysis(end).Results.PostSamples(:,k),50); 
    histogram(post_samples(:,k),50); 
    xline(xopt(:,k), 'LineWidth', 5);
    hold off
    legend('Prior', 'SuS-Samples', 'Posterior', 'Min cost point')
    title(sprintf('Component %d',k))
end
%% Prior domain analysis

%prior_samples = LALOpts.Validation.PriorSamples;

%x1 = linspace(min(prior_samples(:,1)), max(prior_samples(:,1)), 100);
%x2 = linspace(min(prior_samples(:,2)), max(prior_samples(:,2)), 100);

%[x_grid_1, x_grid_2] = meshgrid(x1,x2);
%x_grid = [x_grid_1(:), x_grid_2(:)];

%logL_real = LALOpts.LogLikelihood(x_grid);
%logL_pck = uq_evalModel(pck, x_grid);

%figure
%hold on
%scatter3(LALAnalysis.ExpDesign.X(:,1), LALAnalysis.ExpDesign.X(:,2), LALAnalysis.ExpDesign.LogLikelihood,  "filled")
%surfplot = surf(x1, x2, reshape(logL_real, 100, 100));
%surfplot.EdgeColor = 'none';
%spck = surf(x1,x2,reshape(logL_pck, 100, 100),'FaceAlpha',0.5);
%hold off
%title('Prior domain log-likelihood')
%xlabel('E')
%ylabel('discrepancy')
%zlabel('logL')
%xlim([min(x1), max(x1)])
%ylim([min(x2), max(x2)])
%legend('Experimental design', 'Real logL', 'Surrogate logL')


%% Posterior domain analysis

%x1 = linspace(min(post_samples(:,1)), max(post_samples(:,1)), 100);
%x2 = linspace(min(post_samples(:,2)), max(post_samples(:,2)), 100);

%[x_grid_1, x_grid_2] = meshgrid(x1,x2);
%x_grid = [x_grid_1(:), x_grid_2(:)];

%logL_real = LALOpts.LogLikelihood(x_grid);
%logL_pck = uq_evalModel(pck, x_grid);

%figure
%hold on
%scatter3(LALAnalysis.ExpDesign.X(:,1), LALAnalysis.ExpDesign.X(:,2), LALAnalysis.ExpDesign.LogLikelihood,  "filled")
%surfplot = surf(x1, x2, reshape(logL_real, 100, 100));
%surfplot.EdgeColor = 'none';
%spck = surf(x1,x2,reshape(logL_pck, 100, 100),'FaceAlpha',0.5);
%hold off
%title('Posterior domain log-likelihood')
%xlabel('E')
%ylabel('discrepancy')
%zlabel('logL')
%xlim([min(x1), max(x1)])
%ylim([min(x2), max(x2)])
%legend('Experimental design', 'Real logL', 'Surrogate logL')
