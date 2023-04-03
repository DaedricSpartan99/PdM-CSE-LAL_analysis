% LAL = likelihood active learning

clearvars
rng(100,'twister')
uqlab

addpath('../tools')

%% Input definition

% Prior definition
PriorOpts.Name = 'Prior';
PriorOpts.Marginals(1).Type = 'Gaussian';    % prior form
PriorOpts.Marginals(1).Moments = [0., 1.];   % prior mean and variance
PriorInput = uq_createInput(PriorOpts);

%% Likelihood definition

% peaks position
y = [1.5, -1., -2.5];

% peaks extension
std_disc = [0.1, 0.05, 0.02];

% distance scaling
a = 1.5;

% test singular function
log_likelihood_handle = @(x, Y) max(log(mean(normpdf((Y-a*x) ./ std_disc) ./ std_disc, 2)), -400);
log_likelihood = @(x) max(log(mean(normpdf((y-a*x) ./ std_disc) ./ std_disc, 2)), -400);

%% Analytical solution

post_means = a * y ./ (a^2 + std_disc.^2);
post_std = std_disc ./ sqrt(a^2 + std_disc.^2);

Z = mean(normpdf(y./sqrt(a^2 + std_disc.^2)) ./ sqrt(a^2 + std_disc.^2), 2);
posterior = @(x) mean(normpdf((x - post_means) ./ post_std) .* normpdf(y ./ sqrt(a^2 + std_disc.^2)) ./ std_disc, 2) / Z;


%% Analytical solution plot

% % validation set
% xplot = linspace(-5, 5, 1000);
% 
% % validation
% sum(abs(posterior(xplot') - exp(log_likelihood(xplot')) .* normpdf(xplot') ./ Z))
% trapz(posterior(xplot'), -xplot')
% trapz(exp(log_likelihood(xplot')) .* normpdf(xplot') ./ Z, -xplot')
% 
% figure
% hold on
% plot(xplot, normpdf(xplot), 'DisplayName', 'Prior')
% plot(xplot, posterior(xplot'), 'DisplayName', 'Posterior')
% %plot(xplot, exp(log_likelihood(xplot')) .* normpdf(xplot') ./ Z, 'DisplayName', 'Validation')
% hold off
% title('Prior vs Analytical posterior')
% xlabel('X')
% ylabel('Distribution')
% legend

%% Create experimental design

%init_eval = 10;

%init_X = uq_getSample(PriorInput, init_eval);

% Three peaks, two discovered
%init_X = [-1.47; -1.44; 0.1090; -0.5877; -0.1903; 1.0143; -2.5958; 1.96; -1.96];

% Two peaks
%init_X = [0.1090; -0.5877; -0.1903; 1.0143; -2.5958; -1.1672; 0.4420; 0.9379; -1.0952; 0.1894];

% Three peaks table experiment
init_X = [-1.47; -1.44; 0.11; -1; -0.2; 0.8; -2.6; 1.5; 2; -2; -1.8; -0.5];
%init_X = [-1.47; -1.44; 0.11; -1; -0.2; 0.8; -2.6; 1.5 ; 2; -2; -1.8];
%init_X = [-1.47; -1.44; 0.11; -1; -0.2; 0.8; -2.6; 1.5; 2; -2];

init_logL = log_likelihood(init_X);

%% Initial state

% validation set
xplot = linspace(-4, 4, 1000);

figure
hold on
plt = plot(xplot, log(normpdf(xplot)), '--', 'DisplayName', 'Log-Prior', 'LineWidth',2)
plt.Color = "#EDB120";
plot(xplot', log_likelihood(xplot'), 'b-', 'DisplayName', 'Log-likelihood', 'LineWidth',2)         
plt = scatter(init_X, init_logL, 70, 'Filled', 'DisplayName', 'Experimental design')
plt.MarkerFaceColor = "#77AC30";
plt.MarkerEdgeColor = [0. 0. 0];
plt.LineWidth = 2.;
hold off
%title('Initial state')
xlabel('X')
ylabel('Log-P.D.F')
ylim([-800, 100])
grid on
legend

drawnow

%% Run analysis

clear LALOpts

LALOpts.ExpDesign.X = init_X;
LALOpts.ExpDesign.LogLikelihood = init_logL;

%LALOpts.Bus.logC = -1.9;
%LALOpts.Bus.p0 = 0.1;                            % Quantile probability for Subset
%LALOpts.Bus.BatchSize = 5e4;                             % Number of samples for Subset simulation
%LALOpts.Bus.MaxSampleSize = 1e6;
LALOpts.MaximumEvaluations = 40;
LALOpts.Bus.CStrategy = 'maxpck';
%LALOpts.Delaunay.maxk = 10;
%LALOpts.OptMode = 'single';

LALOpts.MetaOpts.MetaType = 'PCK';
LALOpts.MetaOpts.Mode = 'optimal';   
LALOpts.MetaOpts.Kriging.Optim.Bounds = [0.2; 1];
%LALOpts.MetaOpts.PCE.Degree = 0:2;
%LALOpts.MetaOpts.PCK.Kriging.Optim.Tol = 1e-5;
%LALOpts.MetaOpts.Kriging.Corr.Family = 'gaussian';
%LALOpts.MetaOpts.PCK.Kriging.theta = 9.999;

%LALOpts.MetaOpts.MetaType = 'Kriging';
%LALOpts.MetaOpts.Corr.Family = 'gaussian';
%LALOpts.MetaOpts.Optim.Bounds = [0.05; 1];


%LALOpts.SelectMax = 1;
LALOpts.ClusterRange = 3;

%LALOpts.DBMinPts = 5;

LALOpts.LogLikelihood = log_likelihood;
LALOpts.Prior = PriorInput;

LALOpts.StoreBusResults = true;
%LALOpts.FilterOutliers = false;
%LALOpts.ClusteredMetaModel = true;

LALOpts.Bus.BatchSize = 5000;
LALOpts.Bus.MaxSampleSize = 500000;

LALAnalysis = lal_analysis(LALOpts);


%% Final state plot

% validation set
xplot = linspace(-3.5, 3.5, 1000);

pck = LALAnalysis.BusAnalysis(end).Opts.LogLikelihood;
[lpck_mean, lpck_var] = uq_evalModel(pck,xplot');
lconf = [lpck_mean' + 2*sqrt(lpck_var'), fliplr(lpck_mean' - 2*sqrt(lpck_var'))];

figure

p = fill([xplot, fliplr(xplot)],lconf, [0.6 0.6 0.6], 'FaceAlpha',0.3, 'EdgeColor', 'none', 'DisplayName', 'PCK conf. interval');
%p.Color = [0.5, 0.5, 0.5];
hold on
rplt = plot(xplot', log_likelihood(xplot'), 'DisplayName', 'Real log-likelihood')
rplt.LineWidth = 2.;
rplt.LineStyle = ':';
rplt.Color = [1. 0.2 0.05];%"#EDB120";
pckplt = plot(xplot, lpck_mean, 'DisplayName', 'PCK log-likelihood')
pckplt.LineWidth = 3.;
pckplt.Color = 	"blue";

plt = scatter(LALAnalysis.ExpDesign.X, LALAnalysis.ExpDesign.LogLikelihood, 'Filled', 'DisplayName', 'Enrichement ED points')
plt.MarkerFaceColor = [0.9290 0.8540 0.1250];
plt.MarkerEdgeColor = [0. 0. 0];
plt.LineWidth = 1.;

plt = scatter(init_X, init_logL, 40, 'Filled', 'DisplayName', 'Initial ED')
plt.MarkerFaceColor = "#A2142F";%"#77AC30";
plt.MarkerFaceAlpha = 0.8;
plt.MarkerEdgeColor = [0. 0. 0];

hold off
%title('Final state')
xlabel('X')
ylabel('log-likelihood')
ylim([-800, 100])
grid on
legend()

drawnow

%% Prior vs Posterior plot

prior_plot = normpdf(xplot');

ref_posterior = posterior(xplot');

lal_evidence = LALAnalysis.Evidence;
lal_posterior = exp(lpck_mean + log(prior_plot)) / lal_evidence;

figure
hold on
plot(xplot', prior_plot, 'DisplayName', 'Prior')
plot(xplot', ref_posterior, 'DisplayName', 'Reference posterior')
plot(xplot', lal_posterior, 'DisplayName', 'LAL posterior') 
scatter(LALAnalysis.ExpDesign.X, posterior(LALAnalysis.ExpDesign.X), 'Filled', 'DisplayName', 'Experimental design')
scatter(init_X, posterior(init_X), 'Filled', 'DisplayName', 'Initial experimental design')
hold off
xlabel('X')
ylabel('Probability density')
legend

drawnow

%% Evidence convergence (Weak)

iterations = size(LALAnalysis.BusAnalysis,2);
evidences = zeros(iterations,1);

for i = 1:iterations
    evidences(i) = LALAnalysis.BusAnalysis(i).Results.Evidence;
end

%weak_error = abs(evidences(2:end) - evidences(1:end-1));

figure
scatter(1:iterations, abs(evidences - Z), 'filled')
set(gca, 'YScale', 'log')
xlabel('Iteration')
ylabel('|\hat{\mathcal{Z}}^{(n)} - Z|')
%title('Evidence convergence')

drawnow 

%% Experimental design show

% figure
% histogram(LALAnalysis.ExpDesign.X, ceil(length(LALAnalysis.ExpDesign.X)/2))
% xlabel('X')
% ylabel('Occurrencies')
% title("Experimental design show up")
% 
% drawnow

%% Show LOO error relationship

M = 10000;
h = 10 / M;
xval = linspace(-5, 5, M)';

prior_eval = normpdf(xval);
logL = log_likelihood(xval);
post_eval = exp(logL) .* prior_eval ./ Z;

meanl_prior = trapz(xval, logL .* prior_eval);
varl_prior = trapz(xval, (logL - meanl_prior).^2 .* prior_eval);

meanl_post = trapz(xval, logL .* post_eval);
varl_post = trapz(xval, (logL - meanl_post).^2 .* post_eval);

pck_post_err = zeros(1,iterations);
pck_loo_err = zeros(1,iterations);
pck_prior_err = zeros(1,iterations);

for i = 1:iterations
    dl = uq_evalModel(LALAnalysis.BusAnalysis(i).Opts.LogLikelihood, xval) - logL;

    pck_prior_err(i) = trapz(xval, dl.^2 .* prior_eval) / varl_prior;
    pck_post_err(i) = trapz(xval, dl.^2 .* post_eval) / varl_post;
    pck_loo_err(i) = LALAnalysis.BusAnalysis(i).Opts.LogLikelihood.Error.LOO;
end

% Select range
iters = 5:iterations;
pck_prior_err = pck_prior_err(iters);
pck_post_err = pck_post_err(iters);
pck_loo_err = pck_loo_err(iters);

% Fit errors
[a,~] = polyfit(iters, log10(pck_post_err), 1);
[b,~] = polyfit(iters, log10(pck_loo_err), 1);
[c,~] = polyfit(iters, log10(pck_prior_err), 1);

figure
hold on
plot(iters, 10.^(c(1) .* iters + c(2)), 'DisplayName', 'Prior error Fit', 'Color', "black", 'LineStyle', '-', 'LineWidth', 2)
plot(iters, 10.^(a(1) .* iters + a(2)), 'DisplayName', 'Post error Fit', 'Color', "black", 'LineStyle', ':', 'LineWidth', 2)
plot(iters, 10.^(b(1) .* iters + b(2)), 'DisplayName', 'LOO error Fit', 'Color',"black" , 'LineStyle', '--', 'LineWidth', 2)
scatter(iters, pck_prior_err, 60, 'filled', 'DisplayName', 'Prior validation error', 'MarkerFaceColor', "#EDB120")
scatter(iters, pck_post_err, 60, 'filled', 'DisplayName', 'Posterior validation error', 'MarkerFaceColor', "#7E2F8E")
scatter(iters, pck_loo_err, 60, 'filled', 'DisplayName', 'PCK LOO error', 'MarkerFaceColor', "#77AC30")
hold off
set(gca, 'YScale', 'log')
xlabel('Iteration')
ylabel('Error')
ylim([1e-5, 1e6])
%title('Posterior or prior error vs. LOO error')
grid on
%ylim([min(pck_loo_err) / 10, 1000])
legend('interpreter','latex', 'FontSize', 10, 'NumColumns', 2)


fprintf("Prior validation convergence rate: %f\n", -c(1))
fprintf("Posterior validation convergence rate: %f\n", -a(1))
fprintf("LOO convergence rate: %f\n", -b(1))

fprintf("Prior validation final error: %g\n", pck_prior_err(end))
fprintf("Posterior validation final error: %g\n", pck_post_err(end))
fprintf("LOO convergence rate: %f\n", pck_loo_err(end))

drawnow

%% Execute using MCMC on reference posterior

BayesOpts.Type = 'Inversion';
%BayesOpts.Name = 'User-defined likelihood inversion';
BayesOpts.Prior = PriorInput;
BayesOpts.Data.y = y;
BayesOpts.LogLikelihood = log_likelihood_handle;

refBayesAnalysis = uq_createAnalysis(BayesOpts);

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

%prior_samples = prior_samples(prior_logL_samples > quantile(prior_logL_samples, 0.1), :);
%prior_logL_samples = prior_logL_samples(prior_logL_samples > quantile(prior_logL_samples, 0.1));

%% Execute using MCMC on surrogate final state posterior

BayesOpts.Type = 'Inversion';
%BayesOpts.Name = 'User-defined likelihood inversion';
BayesOpts.Prior = PriorInput;
BayesOpts.Data.y = y;
BayesOpts.LogLikelihood = @(x,Y) uq_evalModel(pck, x);

lalBayesAnalysis = uq_createAnalysis(BayesOpts);

%% post sample exctraction and clean up

M = size(lalBayesAnalysis.Results.PostProc.PostSample,2); % number of time-steps
Solver.MCMC.NChains = lalBayesAnalysis.Internal.Solver.MCMC.NChains;

lal_post_samples = permute(lalBayesAnalysis.Results.PostProc.PostSample, [1, 3, 2]);
lal_post_samples = reshape(lal_post_samples, [], M);
lal_post_logL_samples = reshape(lalBayesAnalysis.Results.PostProc.PostLogLikeliEval, [], 1);

lal_post_samples = lal_post_samples(lal_post_logL_samples > quantile(lal_post_logL_samples, 0.1), :);
lal_post_logL_samples = lal_post_logL_samples(lal_post_logL_samples > quantile(lal_post_logL_samples, 0.1));
lal_post_samples_size = size(lal_post_samples, 1); 

% prepare prior samples
lal_prior_logL_samples = lalBayesAnalysis.LogLikelihood(prior_samples);

%lal_prior_samples = prior_samples(lal_prior_logL_samples > quantile(lal_prior_logL_samples, 0.1), :);
%lal_prior_logL_samples = lal_prior_logL_samples(lal_prior_logL_samples > quantile(lal_prior_logL_samples, 0.1));

%% Run BuS on reference likelihood
BayesOpts.Bus = LALAnalysis.Opts.Bus;
BayesOpts.Bus.logC = -log(7);
BayesOpts.Prior = LALAnalysis.Opts.Prior;

BayesOpts.Bus.BatchSize = 10000;
BayesOpts.Bus.MaxSampleSize = 1000000;

% a= 1.5;
% modelopts.mHandle = @(x) log(mean(normpdf((y-a*x) ./ std_disc) ./ std_disc, 2));
% modelopts.isVectorized = true;
% lModel = uq_createModel(modelopts);
% 
% BayesOpts.LogLikelihood = lModel; 

BayesOpts.LogLikelihood = pck; 

BusAnalysis = bus_analysis(BayesOpts);

px_samples = BusAnalysis.Results.Bus.PostSamples;
lsf = BusAnalysis.Results.Bus.LSF;
[mean_post_LSF,~] = uq_evalModel(lsf, px_samples);
bus_posterior = px_samples(mean_post_LSF <= 0, 2:end);

%% Bayesian analysis samples plot

figure
hold on
histogram(prior_samples,100, 'Normalization', 'pdf');
histogram(post_samples,100, 'Normalization', 'pdf', 'FaceColor', "#EDB120", 'FaceAlpha', 1, 'EdgeColor', 'none'); 
histogram(bus_posterior,100, 'Normalization', 'pdf', 'FaceColor', "#77AC30",'FaceAlpha', 1, 'EdgeColor', 'none'); 
plot(xplot', lal_posterior,'k--','LineWidth',1.5)
hold off
xlabel('X')
ylabel('Probability density function')
grid on
legend('Prior samples', 'MCMC post. samples', 'BuS post. samples', 'Exact posterior', 'FontSize', 11)


figure
hold on
histogram(prior_samples,100, 'Normalization', 'pdf');
histogram(post_samples,100, 'Normalization', 'pdf', 'FaceColor', "#EDB120", 'FaceAlpha', 1, 'EdgeColor', 'none'); 
histogram(lal_post_samples,100, 'Normalization', 'pdf', 'FaceColor', 	"#A2142F", 'EdgeColor', 'none'); 
hold off
xlabel('X')
ylabel('P.d.f')
legend('Prior', 'Ref. MCMC Posterior', 'LAL MCMC Posterior')

%% Posterior validation plot

figure
hold on
hg = qqplot(post_samples, LALAnalysis.PostSamples);
set(hg(1),'marker','o','markersize',3,'markeredgecolor',[0 0 0]);
hg = qqplot(post_samples, lal_post_samples);
set(hg(1),'markersize',3)
hold off
xlabel('Reference MCMC posterior samples')
ylabel('Posterior samples')
legend('LAL MCMC', 'SuS')

%figure
%qqplot(post_samples, lal_post_samples)
%xlabel('Reference MCMC posterior samples')
%ylabel('LAL MCMC posterior samples')

%% Statistical tests

[bus_ks, p_bus_ks] = kstest2(post_samples, LALAnalysis.PostSamples);
[lal_ks, p_lal_ks] = kstest2(post_samples, lal_post_samples);

fprintf('Null hypothesis: same distribution\n')
fprintf('P-value SuS samples: %f, hypothesis rejected: %d\n',p_bus_ks, bus_ks)
fprintf('P-value LAL MCMC samples: %f, hypothesis rejected: %d\n', p_lal_ks, lal_ks)

%% SSLE test

clear BayesOpts
clear Solver

Solver.Type = 'SSLE';

% Expansion options
Solver.SSLE.ExpOptions.Degree = 0:6;

% Experimental design options
Solver.SSLE.ExpDesign.X = LALAnalysis.ExpDesign.X;
Solver.SSLE.ExpDesign.Y = exp(LALAnalysis.ExpDesign.LogLikelihood);

%Solver.SSLE.ExpDesign.NSamples = 1000;
%Solver.SSLE.ExpDesign.NEnrich = 100;

BayesOpts.Type = 'Inversion';
%BayesOpts.Name = 'User-defined likelihood inversion';
BayesOpts.Solver = Solver;
BayesOpts.Prior = PriorInput;
BayesOpts.Data.y = y;
BayesOpts.LogLikelihood = log_likelihood_handle;

ssleBayesAnalysis = uq_createAnalysis(BayesOpts);

%% Display results

uq_postProcessInversion(ssleBayesAnalysis, 'dependence', true)
uq_display(ssleBayesAnalysis)
