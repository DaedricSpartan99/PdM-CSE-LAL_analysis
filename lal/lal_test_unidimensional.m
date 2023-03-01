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

% validation set
xplot = linspace(-5, 5, 1000);

% validation
sum(abs(posterior(xplot') - exp(log_likelihood(xplot')) .* normpdf(xplot') ./ Z))
trapz(posterior(xplot'), -xplot')
trapz(exp(log_likelihood(xplot')) .* normpdf(xplot') ./ Z, -xplot')

figure
hold on
plot(xplot, normpdf(xplot), 'DisplayName', 'Prior')
plot(xplot, posterior(xplot'), 'DisplayName', 'Posterior')
%plot(xplot, exp(log_likelihood(xplot')) .* normpdf(xplot') ./ Z, 'DisplayName', 'Validation')
hold off
title('Prior vs Analytical posterior')
xlabel('X')
ylabel('Distribution')
legend

%% Create experimental design

%init_eval = 6;

%init_X = uq_getSample(PriorInput, init_eval);

init_X = [-1.47; -1.44; 0.1090; -0.5877; -0.1903; 1.0143; -2.5958; 1.96; -1.96];

init_logL = log_likelihood(init_X);

%% Initial state

% validation set
xplot = linspace(-5, 5, 1000);

figure
hold on
plt = plot(xplot, log(normpdf(xplot)), '--', 'DisplayName', 'Prior', 'LineWidth',2)
plt.Color = "#EDB120";
plot(xplot', log_likelihood(xplot'), 'b-', 'DisplayName', 'Log-likelihood', 'LineWidth',2)         
plt = scatter(init_X, init_logL, 70, 'Filled', 'DisplayName', 'Experimental design')
plt.MarkerFaceColor = "#77AC30";
plt.MarkerEdgeColor = [0. 0. 0];
plt.LineWidth = 2.;
hold off
%title('Initial state')
xlabel('X')
ylabel('Distribution')
grid on
legend

drawnow

%% Finalize analysis (new peaks complection)

clear LALOpts

LALOpts.ExpDesign.X = init_X;
LALOpts.ExpDesign.LogLikelihood = init_logL;

%LALOpts.Bus.logC = -1.9;
%LALOpts.Bus.p0 = 0.1;                            % Quantile probability for Subset
%LALOpts.Bus.BatchSize = 5e4;                             % Number of samples for Subset simulation
%LALOpts.Bus.MaxSampleSize = 1e6;
LALOpts.MaximumEvaluations = 60;
LALOpts.Bus.CStrategy = 'maxpck';
%LALOpts.Delaunay.maxk = 10;
LALOpts.OptMode = 'single';

%LALOpts.PCK.PCE.Degree = 1:15;

LALOpts.PCK.Kriging.Optim.Bounds = [0.2; 1];

%LALOpts.PCK.Kriging.Optim.Tol = 1e-5;
LALOpts.PCK.Kriging.Corr.Family = 'gaussian';
%LALOpts.PCK.Kriging.theta = 9.999;

%LALOpts.SelectMax = 1;
%LALOpts.ClusterRange = 2:15;

%LALOpts.PCK.PCE.Method = 'LARS';

LALOpts.LogLikelihood = log_likelihood;
LALOpts.Prior = PriorInput;

LALOpts.StoreBusResults = true;

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
grid on
legend()

drawnow

%% Prior vs Posterior plot

prior_plot = normpdf(xplot');

ref_posterior = posterior(xplot');

lal_evidence = LALAnalysis.BusAnalysis(end).Results.Evidence;
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
ylabel('|Z^n - Z^{n-1}|')
title('Evidence convergence')

drawnow 

%% Squared distance error convergence plot (Strong)

xval = uq_getSample(PriorInput, 5000);

% Filter quantiles
xqU = quantile(xval, 0.975);
xqB = quantile(xval, 0.025);
xval = xval((xval < xqU) & (xval > xqB));

strong_err = zeros(1,iterations);
strong_log_err = zeros(1,iterations);
pck_err = zeros(1,iterations);

for i = 1:iterations
    logL = log_likelihood(xval);
    dl = uq_evalModel(LALAnalysis.BusAnalysis(i).Opts.LogLikelihood, xval) - logL;

    pck_err(i) = mean(dl.^2);
    strong_log_err(i) = mean(exp(2*logL) .* dl.^2);
    strong_err(i) = mean(exp(2*logL) .* (exp(dl) - 1).^2);
end

% Fit errors
[a,~] = polyfit(1:iterations, log10(strong_err), 1);
[b,~] = polyfit(1:iterations, log10(strong_log_err), 1);
[c,~] = polyfit(1:iterations, log10(pck_err), 1);

figure
hold on
plot(1:iterations, 10.^(a(1) .* (1:iterations) + a(2)), 'DisplayName', '$||\hat{L} - L||^2_{\pi}$ Fit')
plot(1:iterations, 10.^(b(1) .* (1:iterations) + b(2)), 'DisplayName', '$||L (\hat{\ell} - \ell)||^2_{\pi}$ Fit')
plot(1:iterations, 10.^(c(1) .* (1:iterations) + c(2)), 'DisplayName', '$||(\hat{\ell} - \ell)||^2_{\pi}$ Fit')
scatter(1:iterations, strong_err, 'filled', 'DisplayName', '$||\hat{L} - L||^2_{\pi}$')
scatter(1:iterations, strong_log_err, 'filled', 'DisplayName', '$||L (\hat{\ell} - \ell)||^2_{\pi}$')
scatter(1:iterations, pck_err, 'filled', 'DisplayName', '$||(\hat{\ell} - \ell)||^2_{\pi}$')
hold off
set(gca, 'YScale', 'log')
xlabel('Iteration')
ylabel('Prior strong error')
title('Prior weighted strong error convergence')
legend('interpreter','latex')

sprintf("Strong prior error convergence rate: %f", -a(1))
sprintf("Expected number of iterations to reduce order of magnitude: %d", ceil(-1.0/a(1)))

sprintf("Strong weighted prior log-error convergence rate: %f", -b(1))

sprintf("Log-likelihood PCK convergence rate: %f", -c(1))

drawnow

%% Experimental design show

figure
histogram(LALAnalysis.ExpDesign.X, ceil(length(LALAnalysis.ExpDesign.X)/2))
xlabel('X')
ylabel('Occurrencies')
title("Experimental design show up")

drawnow

%% Show LOO error relationship

xval = uq_getSample(PriorInput, 10000);

% Filter quantiles
xqU = quantile(xval, 0.975);
xqB = quantile(xval, 0.025);
xval = xval((xval < xqU) & (xval > xqB));

pck_post_err = zeros(1,iterations);
pck_loo_err = zeros(1,iterations);
pck_prior_err = zeros(1,iterations);

for i = 1:iterations
    logL = log_likelihood(xval);
    dL = exp(uq_evalModel(LALAnalysis.BusAnalysis(i).Opts.LogLikelihood, xval)) - exp(logL);

    pck_prior_err(i) = mean(dL.^2);
    pck_post_err(i) = mean(exp(logL) .* dL.^2 ./ Z);
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
%title('Posterior or prior error vs. LOO error')
grid on
ylim([min(pck_loo_err) / 10, 1000])
legend('interpreter','latex', 'FontSize', 20)

sprintf("Prior validation convergence rate: %f", -c(1))
sprintf("Posterior validation convergence rate: %f", -a(1))
sprintf("LOO convergence rate: %f", -b(1))

drawnow

%% Execute using MCMC

BayesOpts.Type = 'Inversion';
BayesOpts.Name = 'User-defined likelihood inversion';
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

prior_samples = prior_samples(prior_logL_samples > quantile(prior_logL_samples, 0.1), :);
prior_logL_samples = prior_logL_samples(prior_logL_samples > quantile(prior_logL_samples, 0.1));

%% Limit state function plot

xopt = LALAnalysis.OptPoints(end).X;
logL_PCK = LALAnalysis.PCK(end);

figure
hold on
histogram(prior_samples(:,1),50);
histogram(post_samples(:,1),50); 
histogram(LALAnalysis.BusAnalysis(end).Results.PostSamples(:,1),50); 
xline(xopt(:,1), 'LineWidth', 5);
hold off
xlabel('X')
ylabel('Occurences')
legend('Prior', 'Posterior', 'SuS-Samples', 'Min cost point')

%figure
%hold on
%surfplot = surf(xplot, pplot, lsf_eval);
%surfplot.EdgeColor = 'none';
%surfplot.FaceAlpha = 0.5;
%scatter3(lsf_X, lsf_P, lsf_samples_mean, 'Filled')
%plot3([lsf_X(:),lsf_X(:)]', [lsf_P(:),lsf_P(:)]', [-lsf_samples_std(:),lsf_samples_std(:)]'+lsf_samples_mean(:)', '-r') 
%hold off
%xlabel('X')
%ylabel('P')
%zlabel('LSF')
%title('Limit state function')
%legend('LSF', 'Means', 'Std');
