% LAL = likelihood active learning

clearvars
rng(100,'twister')
uqlab

%% Input definition

% Prior definition
PriorOpts.Name = 'Prior';
PriorOpts.Marginals(1).Type = 'Gaussian';    % prior form
PriorOpts.Marginals(1).Moments = [0., 1.];   % prior mean and variance
PriorInput = uq_createInput(PriorOpts);

%% Likelihood definition

% test singular function
log_likelihood = @(x) log(abs(sin(4*(x+eps) - 4)./(4*(x+eps) - 4)));

%% Create experimental design

init_eval = 20;

LALOpts.ExpDesign.X = uq_getSample(PriorInput, init_eval);
LALOpts.ExpDesign.LogLikelihood = log_likelihood(LALOpts.ExpDesign.X);

init_X = LALOpts.ExpDesign.X;
init_logL = LALOpts.ExpDesign.LogLikelihood;

%% Initial state

% validation set
xplot = linspace(-5, 5, 1000);

figure
scatter(LALOpts.ExpDesign.X, LALOpts.ExpDesign.LogLikelihood, 'Filled', 'DisplayName', 'Experimental design')
hold on
plot(xplot, log_likelihood(xplot), 'DisplayName', 'Real log-likelihood')         
hold off
title('Initial state')
xlabel('X')
ylabel('log-likelihood')
legend

drawnow


%% Run Analysis

LALOpts.Bus.logC = 0.; % best value: -max log(L) 
LALOpts.Bus.p0 = 0.1;                            % Quantile probability for Subset
LALOpts.Bus.BatchSize = 1e3;                             % Number of samples for Subset simulation
%LALOpts.Bus.MaxSampleSize = 1e4;
LALOpts.MaximumEvaluations = 100;

%LALOpts.PCK.PCE.Degree = 0:2:12;
LALOpts.PCK.PCE.Method = 'LARS';
%LALOpts.PCK.Optim.Method = 'CMAES';
%LALOpts.PCK.Kriging.Corr.Type = 'Separable';
%LALOpts.PCK.Kriging.Optim.MaxIter = 1000;
%LALOpts.PCK.Kriging.Corr.Family = 'Gaussian';
%LALOpts.PCK.Display = 'verbose';

LALOpts.LogLikelihood = log_likelihood;
LALOpts.Prior = PriorInput;

LALOpts.StoreBusResults = true;

LALAnalysis = lal_analysis(LALOpts);

%% Final state plot

pck = LALAnalysis.BusAnalysis(end).Opts.LogLikelihood;
[lpck_mean, lpck_var] = uq_evalModel(pck,xplot');
lconf = [lpck_mean' + 2*sqrt(lpck_var'), fliplr(lpck_mean' - 2*sqrt(lpck_var'))];

figure

p = fill([xplot, fliplr(xplot)],lconf,'red', 'FaceAlpha',0.3, 'EdgeColor', 'none');
hold on
plot(xplot, log_likelihood(xplot), 'DisplayName', 'Real log-likelihood')
plot(xplot, lpck_mean, 'DisplayName', 'Surrogate log-likelihood')
scatter(LALAnalysis.ExpDesign.X, LALAnalysis.ExpDesign.LogLikelihood, 'Filled', 'DisplayName', 'Experimental design')
scatter(init_X, init_logL, 'Filled', 'DisplayName', 'Initial experimental design')
hold off
title('Final state')
xlabel('X')
ylabel('log-likelihood')
legend()

%% Prior vs Posterior plot

prior_plot = normpdf(xplot);

ref_posterior = exp(log_likelihood(xplot) + log(prior_plot));
ref_evidence = trapz(xplot, ref_posterior);
ref_posterior = ref_posterior / ref_evidence;

lal_evidence = LALAnalysis.BusAnalysis(end).Results.Evidence;
lal_posterior = exp(lpck_mean' + log(prior_plot)) / lal_evidence;

figure
hold on
plot(xplot, prior_plot, 'DisplayName', 'Prior')
plot(xplot, ref_posterior, 'DisplayName', 'Reference posterior')
plot(xplot, lal_posterior, 'DisplayName', 'LAL posterior') 
scatter(LALAnalysis.ExpDesign.X, exp(LALAnalysis.ExpDesign.LogLikelihood + log(normpdf(LALAnalysis.ExpDesign.X))) / ref_evidence, 'Filled', 'DisplayName', 'Experimental design')
scatter(init_X, exp(init_logL + log(normpdf(init_X))) / ref_evidence, 'Filled', 'DisplayName', 'Initial experimental design')
hold off
xlabel('X')
ylabel('Probability density')
legend

%% Evidence convergence (Weak)

iterations = size(LALAnalysis.BusAnalysis,2);
evidences = zeros(iterations,1);

for i = 1:iterations
    evidences(i) = LALAnalysis.BusAnalysis(i).Results.Evidence;
end

weak_error = abs(evidences(2:end) - evidences(1:end-1));

figure
scatter(2:iterations, weak_error, 'filled')
set(gca, 'YScale', 'log')
xlabel('Iteration')
ylabel('|Z^n - Z^{n-1}|')
title('Evidence convergence')


%% Squared distance error convergence plot (Strong)

xval = uq_getSample(PriorInput, 5000);
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

%% Experimental design show

figure
histogram(LALAnalysis.ExpDesign.X, length(LALAnalysis.ExpDesign.X)/3)
xlabel('X')
ylabel('Occurrencies')
title("Experimental design show up")

%% Show LOO error relationship

xval = uq_getSample(PriorInput, 5000);
pck_post_err = zeros(1,iterations);
pck_loo_err = zeros(1,iterations);
pck_prior_err = zeros(1,iterations);

for i = 1:iterations
    logL = log_likelihood(xval);
    dl = uq_evalModel(LALAnalysis.BusAnalysis(i).Opts.LogLikelihood, xval) - logL;

    pck_prior_err(i) = mean(dl.^2);
    pck_post_err(i) = mean(exp(logL) .* dl.^2);
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
title('Posterior or prior error vs. LOO error')
legend('interpreter','latex')

sprintf("Prior validation convergence rate: %f", -c(1))
sprintf("Posterior validation convergence rate: %f", -a(1))
sprintf("LOO convergence rate: %f", -b(1))

%% Limit state function plot

lsf = LALAnalysis.BusAnalysis(1).Results.Bus.LSF;
lsf_samples = LALAnalysis.BusAnalysis(1).Results.Bus.PostSamples;
lsf_X = lsf_samples(:,2);
lsf_P = lsf_samples(:,1);
[lsf_samples_mean, lsf_samples_var] = uq_evalModel(lsf, lsf_samples);
lsf_samples_std = sqrt(lsf_samples_var);

xplot = linspace(min(lsf_X), max(lsf_X), 100);
pplot = linspace(0,1, 100);

[x_grid_1, x_grid_2] = meshgrid(pplot,xplot);
px_grid = [x_grid_1(:), x_grid_2(:)];

lsf_eval = uq_evalModel(lsf, px_grid);
lsf_eval = reshape(lsf_eval, 100, 100);

figure
hold on
surfplot = surf(xplot, pplot, lsf_eval);
surfplot.EdgeColor = 'none';
surfplot.FaceAlpha = 0.5;
scatter3(lsf_X, lsf_P, lsf_samples_mean, 'Filled')
plot3([lsf_X(:),lsf_X(:)]', [lsf_P(:),lsf_P(:)]', [-lsf_samples_std(:),lsf_samples_std(:)]'+lsf_samples_mean(:)', '-r') 
hold off
xlabel('X')
ylabel('P')
zlabel('LSF')
title('Limit state function')
legend('LSF', 'Means', 'Std');
