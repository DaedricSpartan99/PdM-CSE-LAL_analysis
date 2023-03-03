clearvars
rng(100,'twister')
uqlab

addpath('../lal')
addpath('../tools')

%% Data set
load('uq_Example_BayesianPreyPred.mat')

%% Model definition

normYear = Data.year-Data.year(1);

ModelOpts.mHandle = @(x) uq_predatorPreyModel(x,normYear);
ModelOpts.isVectorized = true;

myForwardModel = uq_createModel(ModelOpts);

%% Prior setup

PriorOpts.Marginals(1).Name = ('alpha');
PriorOpts.Marginals(1).Type = 'LogNormal';
PriorOpts.Marginals(1).Moments = [1 0.1];
%PriorOpts.Marginals(1).Type = 'Constant';
%PriorOpts.Marginals(1).Parameters = 0.5524;

PriorOpts.Marginals(2).Name = ('beta');
PriorOpts.Marginals(2).Type = 'LogNormal';
PriorOpts.Marginals(2).Moments = [0.05 0.005];
%PriorOpts.Marginals(2).Type = 'Constant';
%PriorOpts.Marginals(2).Parameters = 0.0309;

PriorOpts.Marginals(3).Name = ('gamma');
PriorOpts.Marginals(3).Type = 'LogNormal';
PriorOpts.Marginals(3).Moments = [1 0.1];
%PriorOpts.Marginals(3).Type = 'Constant';
%PriorOpts.Marginals(3).Parameters = 0.8929;

PriorOpts.Marginals(4).Name = ('delta');
PriorOpts.Marginals(4).Type = 'LogNormal';
PriorOpts.Marginals(4).Moments = [0.05 0.005];
%PriorOpts.Marginals(4).Type = 'Constant';
%PriorOpts.Marginals(4).Parameters = 0.0279;

PriorOpts.Marginals(5).Name = ('initH');
PriorOpts.Marginals(5).Type = 'LogNormal';
PriorOpts.Marginals(5).Parameters = [log(10) 1];
%PriorOpts.Marginals(4).Type = 'Constant';
%PriorOpts.Marginals(4).Parameters = 32.2464;

PriorOpts.Marginals(6).Name = ('initL');
PriorOpts.Marginals(6).Type = 'LogNormal';
PriorOpts.Marginals(6).Parameters = [log(10) 1];
%PriorOpts.Marginals(6).Type = 'Constant';
%PriorOpts.Marginals(6).Parameters = 4.5109;

myPriorDist = uq_createInput(PriorOpts);

%% Measurements

myData(1).y = Data.hare.'/1000; %in 1000
myData(1).Name = 'Hare data';
myData(1).MOMap = 1:21; % Output ID

myData(2).y = Data.lynx.'/1000; %in 1000
myData(2).Name = 'Lynx data';
myData(2).MOMap = 22:42; % Output ID

%% Discrepancy model

SigmaOpts.Marginals(1).Name = 'Sigma2L';
SigmaOpts.Marginals(1).Type = 'Lognormal';
SigmaOpts.Marginals(1).Parameters = [-1 1];

SigmaDist1 = uq_createInput(SigmaOpts);

SigmaOpts.Marginals(1).Name = 'Sigma2H';
SigmaOpts.Marginals(1).Type = 'Lognormal';
SigmaOpts.Marginals(1).Parameters = [-1 1];

SigmaDist2 = uq_createInput(SigmaOpts);

% Choose otion
DiscrepancyOpts(1).Type = 'Gaussian';
%DiscrepancyOpts(1).Prior = SigmaDist1;
DiscrepancyOpts(1).Parameters = 12.4961;
DiscrepancyOpts(2).Type = 'Gaussian';
%DiscrepancyOpts(2).Prior = SigmaDist2;
DiscrepancyOpts(2).Parameters = 8.4746;


%% Reference model definition

Solver.Type = 'MCMC';
Solver.MCMC.Sampler = 'AIES';
Solver.MCMC.Steps = 400;
Solver.MCMC.NChains = 100;

%Solver.MCMC.Visualize.Parameters = [5 6];
%Solver.MCMC.Visualize.Interval = 40;

BayesOpts.Type = 'Inversion';
BayesOpts.Name = 'Bayesian model';
BayesOpts.Prior = myPriorDist;
BayesOpts.Data = myData;
BayesOpts.Discrepancy = DiscrepancyOpts;
BayesOpts.Solver = Solver;

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

init_eval = 50;
log_likelihood = refBayesAnalysis.LogLikelihood;

LALOpts.ExpDesign.X = uq_getSample(refBayesAnalysis.Internal.FullPrior, init_eval, 'Sobol');
LALOpts.ExpDesign.LogLikelihood = log_likelihood(LALOpts.ExpDesign.X);

init_X = LALOpts.ExpDesign.X;
init_logL = LALOpts.ExpDesign.LogLikelihood;

%qinit = quantile(init_logL, 0.05);
%init_X = init_X(init_logL > qinit,:);
%init_logL = init_logL(init_logL > qinit);

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


%% Bayesian analysis (tuning first peaks step)

clear LALOpts

%LALOpts.Bus.logC = 300; %-max(post_logL_samples); % best value: -max log(L) 
%LALOpts.Bus.p0 = 0.1;                            % Quantile probability for Subset
%LALOpts.Bus.BatchSize = 1e3;                             % Number of samples for Subset simulation
%LALOpts.Bus.MaxSampleSize = 1e4;
LALOpts.MaximumEvaluations = 100;
LALOpts.ExpDesign.X = init_X;
LALOpts.ExpDesign.LogLikelihood = init_logL;
LALOpts.PlotLogLikelihood = true;
LALOpts.Bus.CStrategy = 'maxpck';
%LALOpts.Bus.Delaunay.maxk = 50;
%LALOpts.OptMode = 'single';
 
LALOpts.SelectMax = 1;
LALOpts.ClusterRange = 2:15;

LALOpts.MetaOpts.MetaType = 'PCK';
LALOpts.MetaOpts.PCK.PCE.Degree = 0:2;
%LALOpts.MetaOpts.PCK.Mode = 'optimal';   
LALOpts.MetaOpts.PCK.Kriging.Optim.Bounds = [0.1; 2];
LALOpts.MetaOpts.PCK.Kriging.Corr.Family = 'Gaussian';

%LALOpts.PCK.Kriging.Optim.Bounds = [0.1; 100];

%LALOpts.PCK.PCE.Degree = 1:10;
%LALOpts.PCK.PCE.PolyTypes = {'Hermite', 'Hermite'};
%LALOpts.PCK.Optim.Method = 'CMAES';
%LALOpts.PCK.Kriging.Optim.MaxIter = 500;
%LALOpts.PCK.Kriging.Optim.Tol = 1e-5;
%LALOpts.PCK.Kriging.Corr.Family = 'gaussian';
%LALOpts.PCK.Kriging.Corr.Family = 'matern-5_2';
%LALOpts.PCK.Kriging.Corr.Type = 'separable';
%LALOpts.PCK.Kriging.Corr.Type = 'ellipsoidal';
%LALOpts.PCK.Kriging.Corr.Nugget = 1e-9;
%LALOpts.PCK.Display = 'verbose';

LALOpts.cleanQuantile = 0.025;



LALOpts.LogLikelihood = refBayesAnalysis.LogLikelihood;
LALOpts.Prior = refBayesAnalysis.Internal.FullPrior;

% TODO: cross validate
%LALOpts.Ridge = 0.0;

LALOpts.Bus.BatchSize = 5000;
LALOpts.Bus.MaxSampleSize = 500000;

LALOpts.Validation.PostSamples = post_samples;
LALOpts.Validation.PostLogLikelihood = post_logL_samples;
LALOpts.Validation.PriorSamples = prior_samples;
LALOpts.Validation.PriorLogLikelihood = prior_logL_samples;

LALOpts.StoreBusResults = true;

LALOpts.DBMinPts = 5;

LALOpts.FilterOutliers = false;
LALOpts.ClusteredMetaModel = true;

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
title('Real likelihood visualization of component 5 and 6')

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
title('Surrogate PCK likelihood visualization of component 5 and 6')

drawnow

%% Delaunay steps

clear LALOpts

%LALOpts.Bus.logC = 300; %-max(post_logL_samples); % best value: -max log(L) 
%LALOpts.Bus.p0 = 0.1;                            % Quantile probability for Subset
%LALOpts.Bus.BatchSize = 1e3;                             % Number of samples for Subset simulation
%LALOpts.Bus.MaxSampleSize = 1e4;
LALOpts.MaximumEvaluations = 5;
LALOpts.ExpDesign = FirstLALAnalysis.ExpDesign;
LALOpts.PlotLogLikelihood = true;
%LALOpts.Bus.CStrategy = 'maxpck';
LALOpts.Bus.Delaunay.maxk = 10;
 
LALOpts.OptMode = 'single';
%LALOpts.SelectMax = 2;
%LALOpts.ClusterRange = 2:20;

LALOpts.PCK.Kriging.Optim.Bounds = [0.1; 100];


LALOpts.PCK.PCE.Degree = 1:10;

%LALOpts.PCK.PCE.Degree = 1:15;
%LALOpts.PCK.PCE.PolyTypes = {'Hermite', 'Hermite'};
%LALOpts.PCK.Optim.Method = 'CMAES';
%LALOpts.PCK.Kriging.Optim.MaxIter = 1000;
LALOpts.PCK.Kriging.Corr.Family = 'gaussian';
%LALOpts.PCK.Kriging.Corr.Family = 'Matern-3_2';
%LALOpts.PCK.Kriging.Corr.Type = 'Separable';
LALOpts.PCK.Kriging.Corr.Type = 'ellipsoidal';
%LALOpts.PCK.Kriging.theta = 9.999;
%LALOpts.PCK.Display = 'verbose';

%LALOpts.cleanQuantile = 0.025;

LALOpts.LogLikelihood = refBayesAnalysis.LogLikelihood;
LALOpts.Prior = refBayesAnalysis.Internal.FullPrior;

% TODO: cross validate
%LALOpts.Ridge = 0.0;

LALOpts.Bus.BatchSize = 10000;
LALOpts.Bus.MaxSampleSize = 1000000;

LALOpts.Validation.PostSamples = post_samples;
LALOpts.Validation.PostLogLikelihood = post_logL_samples;
LALOpts.Validation.PriorSamples = prior_samples;
LALOpts.Validation.PriorLogLikelihood = prior_logL_samples;

LALOpts.StoreBusResults = true;

DelaunayLALAnalysis = lal_analysis(LALOpts);

fprintf("---   LogC: %f\n", FirstLALAnalysis.logC(end));
fprintf("---   Found point with likelihood: %f\n", FirstLALAnalysis.OptPoints(end).logL)


xopt = DelaunayLALAnalysis.OptPoints(end).X;
logL_PCK = DelaunayLALAnalysis.PCK(end);

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
title('Real likelihood visualization of component 5 and 6')

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
title('Surrogate PCK likelihood visualization of component 5 and 6')

drawnow


%% Finalize

clear LALOpts

LALOpts.Bus.logC = 300; %-max(post_logL_samples); % best value: -max log(L) 
%LALOpts.Bus.p0 = 0.1;                            % Quantile probability for Subset
%LALOpts.Bus.BatchSize = 1e3;                             % Number of samples for Subset simulation
%LALOpts.Bus.MaxSampleSize = 1e4;
LALOpts.MaximumEvaluations = 20;
LALOpts.ExpDesign = DelaunayLALAnalysis.ExpDesign;
LALOpts.PlotLogLikelihood = true;
%LALOpts.Bus.CStrategy = 'maxpck';
%LALOpts.Bus.Delaunay.maxk = 60;

%LALOpts.Bus.BatchSize = 10000;
%LALOpts.Bus.MaxSampleSize = 1000000;

LALOpts.SelectMax = 2;
LALOpts.ClusterRange = 2;

LALOpts.PCK.Kriging.Optim.Bounds = [0.1; 100];
LALOpts.PCK.PCE.Degree = 1:10;

%LALOpts.PCK.PCE.PolyTypes = {'Hermite', 'Hermite'};
%LALOpts.PCK.Optim.Method = 'CMAES';
%LALOpts.PCK.Kriging.Optim.MaxIter = 1000;
LALOpts.PCK.Kriging.Corr.Family = 'gaussian';
%LALOpts.PCK.Kriging.Corr.Family = 'Matern-3_2';
%LALOpts.PCK.Kriging.Corr.Type = 'Separable';
LALOpts.PCK.Kriging.Corr.Type = 'ellipsoidal';
%LALOpts.PCK.Kriging.theta = 9.999;
%LALOpts.PCK.Display = 'verbose';

%LALOpts.cleanQuantile = 0.025;

LALOpts.LogLikelihood = refBayesAnalysis.LogLikelihood;
LALOpts.Prior = refBayesAnalysis.Internal.FullPrior;

% TODO: cross validate
%LALOpts.Ridge = 0.0;

LALOpts.Bus.BatchSize = 10000;
LALOpts.Bus.MaxSampleSize = 1000000;

LALOpts.Validation.PostSamples = post_samples;
LALOpts.Validation.PostLogLikelihood = post_logL_samples;
LALOpts.Validation.PriorSamples = prior_samples;
LALOpts.Validation.PriorLogLikelihood = prior_logL_samples;

LALOpts.StoreBusResults = true;

LALAnalysis = lal_analysis(LALOpts);

fprintf("---   LogC: %f\n", LALAnalysis.logC(end));
fprintf("---   Found point with likelihood: %f\n", LALAnalysis.OptPoints(end).logL)


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
scatter(ax, init_Xq(:,a1), init_Xq(:,a2), 25, init_logLq,  'filled')
%surfplot.EdgeColor = 'none';
%surfplot.FaceAlpha = 0.5;
%surfplot_pck = surf(Hplot, Lplot, logL_PCK_grid);
%surfplot_pck.EdgeColor = 'none';
%surfplot_pck.FaceAlpha = 0.8;
hold off
title('Real likelihood visualization of component 5 and 6')

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
title('Surrogate PCK likelihood visualization of component 5 and 6')

drawnow



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

%% Analysis: plot of experimental design and real log-likelihood on marginal 5 and 6


%figure
%xlabel('X_5')
%ylabel('X_6')
%zlabel('logL')
%hold on
%scatter3(LALAnalysis.ExpDesign.X(:,5), LALAnalysis.ExpDesign.X(:,6), LALAnalysis.ExpDesign.LogLikelihood ./ LOpts.Parameters.Amplification , "filled")
%surfplot = surf(X5_samples, X6_samples, reshape(logL_real, n_samples, n_samples));
%surfplot.EdgeColor = 'none';
%hold off
%legend('Experimental design', 'Real Log-Likelihood')

