clearvars
rng(100,'twister')
uqlab

addpath('../lal')
addpath('../tools')

%% visualize

uq_figure
[I,~] = imread('SimplySupportedBeam.png');
image(I)
axis equal
set(gca, 'visible', 'off')

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
%PriorOpts.Marginals(5).Type = 'Constant';
%PriorOpts.Marginals(5).Parameters = 1.2317e+04;           % (N/m)
PriorOpts.Marginals(5).Type = 'Gaussian';
PriorOpts.Marginals(5).Moments = [12000 600]; % (N/m)

myPriorDist = uq_createInput(PriorOpts);

%% Measurement setup

myData.y = [12.84; 13.12; 12.13; 12.19; 12.67]/1000; % (m)
myData.Name = 'Mid-span deflection';

DiscrepancyOpts(1).Type = 'Gaussian';
DiscrepancyOpts(1).Parameters = 1e-5;%var(myData.y);

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

init_eval = 70;
log_likelihood = refBayesAnalysis.LogLikelihood;

LALOpts.ExpDesign.X = uq_getSample(refBayesAnalysis.Internal.FullPrior, init_eval);
LALOpts.ExpDesign.LogLikelihood = log_likelihood(LALOpts.ExpDesign.X);

% add extremely small value for third parameter
% take random points and assign a greedy analytical value
disc_guess = min(LALOpts.ExpDesign.X(:,3)) / 10.;

[~,greedy_ind] = maxk(LALOpts.ExpDesign.LogLikelihood,10);
x_greedy = LALOpts.ExpDesign.X(greedy_ind,:);
x_greedy(:,3) = disc_guess;
logL_greedy = log_likelihood(x_greedy);

ql = quantile(LALOpts.ExpDesign.LogLikelihood, 0.05);

LALOpts.ExpDesign.X = [LALOpts.ExpDesign.X; x_greedy];
LALOpts.ExpDesign.LogLikelihood = [LALOpts.ExpDesign.LogLikelihood; logL_greedy];

% remove incredibly small points
LALOpts.ExpDesign.X = LALOpts.ExpDesign.X(LALOpts.ExpDesign.LogLikelihood > ql,:);
LALOpts.ExpDesign.LogLikelihood = LALOpts.ExpDesign.LogLikelihood(LALOpts.ExpDesign.LogLikelihood > ql);

init_X = LALOpts.ExpDesign.X;
init_logL = LALOpts.ExpDesign.LogLikelihood;


%% Bayesian analysis (Analysis step)

clear LALOpts

%LALOpts.Bus.logC = -32; %-max(post_logL_samples); % best value: -max log(L) 
%LALOpts.Bus.p0 = 0.1;                            % Quantile probability for Subset
%LALOpts.Bus.BatchSize = 1e3;                             % Number of samples for Subset simulation
%LALOpts.Bus.MaxSampleSize = 1e4;
LALOpts.MaximumEvaluations = 20;
LALOpts.ExpDesign.X = init_X;
LALOpts.ExpDesign.LogLikelihood = init_logL;
LALOpts.PlotLogLikelihood = true;
LALOpts.Bus.CStrategy = 'maxpck';
%LALOpts.MinCostSamples = 10;  
%LALOpts.SelectMax = 1;
LALOpts.ClusterRange = 3;

%LALOpts.MetaOpts.MetaType = 'Kriging';
%LALOpts.MetaOpts.Optim.Bounds = [0.2; 2];
%LALOpts.MetaOpts.Optim.MaxIter = 100;
%LALOpts.MetaOpts.Corr.Family = 'Gaussian';
%LALOpts.MetaOpts.Trend.Type = 'linear';

LALOpts.MetaOpts.MetaType = 'PCK';
LALOpts.MetaOpts.PCE.Degree = 0:2;
LALOpts.MetaOpts.Mode = 'optimal';   
LALOpts.MetaOpts.Kriging.Optim.Bounds = [0.2, 0.2, 0.01; 100, 100, 2];
LALOpts.MetaOpts.Kriging.Corr.Family = 'gaussian';

%LALOpts.PCK.PCE.PolyTypes = {'Hermite', 'Hermite'};
%LALOpts.PCK.Optim.Method = 'CMAES';
%LALOpts.PCK.Kriging.Optim.MaxIter = 1000;
%LALOpts.PCK.Kriging.Corr.Family = 'Matern-3_2';
%LALOpts.PCK.Kriging.Corr.Type = 'Separable';
%LALOpts.PCK.Kriging.Corr.Type = 'ellipsoidal';
%LALOpts.PCK.Kriging.theta = 9.999;
%LALOpts.PCK.Display = 'verbose';

LALOpts.cleanQuantile = 0.025;
%LALOpts.GradientCost = true;

LALOpts.Bus.BatchSize = 5000;
LALOpts.Bus.MaxSampleSize = 500000;

LALOpts.LogLikelihood = refBayesAnalysis.LogLikelihood;
LALOpts.Prior = refBayesAnalysis.Internal.FullPrior;

LALOpts.DBMinPts = 5;

%LALOpts.FilterOutliers = false;
LALOpts.ClusteredMetaModel = true;

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

%% Scatter validation

% Posterior validation
check_interval = [min(post_logL_samples), max(post_logL_samples)];

figure
hold on
plot(check_interval, check_interval);
scatter(post_logL_samples, uq_evalModel(logL_PCK, post_samples));
hold off
%title('Posterior samples')
ylabel('Surrogate Log-Likelihood')
xlabel('Real Log-Likelihood')
xlim(check_interval)
ylim(check_interval)

% Prior validation
check_interval = [min(prior_logL_samples), max(prior_logL_samples)];

figure
hold on
plot(check_interval, check_interval);
scatter(prior_logL_samples, uq_evalModel(logL_PCK, prior_samples), 'MarkerEdgeColor', '#7E2F8E');
hold off
%title('Posterior samples')
ylabel('Surrogate Log-Likelihood')
xlabel('Real Log-Likelihood')
xlim(check_interval)
ylim(check_interval)

%% Gplot and histograms

X = LALAnalysis.ExpDesign.X;
group = {'Ref. post. samples', 'Initial ED points','Enriched ED points'};
color = lines(3);
TT = [post_samples; X];
idx = [zeros(post_samples_size,1); 2 * ones(size(init_X,1),1); ones(size(X,1) - size(init_X,1),1)];
labeledGroups = categorical(idx, [0 2 1], group);

xnames = {'Load', 'Young modulus', 'Discrepancy variance'};

figure
gplotmatrix(TT, [] ,labeledGroups,color,'.oo',[],[],'grpbars',xnames)

exportgraphics(gcf,'../../final_results/ssb/gplotmatrix.eps')%,'ContentType','vector')

drawnow

%% Optimize plot

h = findobj('Tag','legend');
set(h.EntryContainer.NodeChildren(1).Icon.Transform.Children.Children, 'Style', 'point')
set(h.EntryContainer.NodeChildren(2).Icon.Transform.Children.Children, 'Style', 'point')
set(h.EntryContainer.NodeChildren(3).Icon.Transform.Children.Children, 'Style', 'point')
set(h.EntryContainer.NodeChildren(1).Icon.Transform.Children.Children, 'Size', 15)
set(h.EntryContainer.NodeChildren(2).Icon.Transform.Children.Children, 'Size', 15)
set(h.EntryContainer.NodeChildren(3).Icon.Transform.Children.Children, 'Size', 15)



%% Plot of prior validation errors over last run

iterations = LALOpts.MaximumEvaluations;

pck_post_err = zeros(1,iterations);
pck_loo_err = zeros(1,iterations);
pck_prior_err = zeros(1,iterations);

for i = 1:iterations
    dl_prior = abs(uq_evalModel(LALAnalysis.BusAnalysis(i).Opts.LogLikelihood, prior_samples) - prior_logL_samples);
    dl_post = abs(uq_evalModel(LALAnalysis.BusAnalysis(i).Opts.LogLikelihood, post_samples) - post_logL_samples);

    pck_prior_err(i) = mean(dl_prior.^2) / var(prior_logL_samples);
    %pck_post_err(i) = mean(exp(prior_logL_samples) .* dl.^2);
    pck_post_err(i) = mean(dl_post.^2) / var(post_logL_samples);
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
