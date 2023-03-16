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



%% Run different initial configurations

clear LALOpts

n_runs = 100;
init_eval = 10; 
lal_iter = 40;

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

pck_post_err = zeros(n_runs,lal_iter);
pck_loo_err = zeros(n_runs,lal_iter);
pck_prior_err = zeros(n_runs,lal_iter);

weak_err = zeros(n_runs, lal_iter);

for i = 1:n_runs

    init_X = uq_getSample(PriorInput, init_eval);
    init_logL = log_likelihood(init_X);

    LALOpts.ExpDesign.X = init_X;
    LALOpts.ExpDesign.LogLikelihood = init_logL;

    LALOpts.MaximumEvaluations = lal_iter;
    LALOpts.Bus.CStrategy = 'maxpck';

    LALOpts.MetaOpts.MetaType = 'PCK';
    LALOpts.MetaOpts.Mode = 'optimal';   
    LALOpts.MetaOpts.Kriging.Optim.Bounds = [0.2; 1];
    LALOpts.MetaOpts.PCE.Degree = 0:2;

    LALOpts.LogLikelihood = log_likelihood;
    LALOpts.Prior = PriorInput;

    LALOpts.StoreBusResults = true;

    LALOpts.Bus.BatchSize = 5000;
    LALOpts.Bus.MaxSampleSize = 500000;

    try
        LALAnalysis = lal_analysis(LALOpts);
    catch
        continue;
    end

    % Evidence convergence (Weak)

    iterations = size(LALAnalysis.BusAnalysis,2);

    for j = 1:iterations
        weak_err(i,j) = abs(LALAnalysis.BusAnalysis(j).Results.Evidence - Z);
    end
    
    % Other errors
    for j = 1:iterations
        dl = uq_evalModel(LALAnalysis.BusAnalysis(j).Opts.LogLikelihood, xval) - logL;

        pck_prior_err(i,j) = trapz(xval, dl.^2 .* prior_eval) / varl_prior;
        pck_post_err(i,j) = trapz(xval, dl.^2 .* post_eval) / varl_post;
        pck_loo_err(i,j) = LALAnalysis.BusAnalysis(j).Opts.LogLikelihood.Error.LOO;
    end
    
end

%% Save matrixes

writematrix(pck_prior_err,'pck_prior_err.csv');
writematrix(pck_post_err,'pck_post_err.csv');
writematrix(pck_loo_err,'pck_loo_err.csv');
writematrix(weak_err,'weak_err.csv');

%% Filter out divergent cases from post validation error

val = pck_post_err(:,end);
zval = zscore(val);

% pck_prior_err = pck_prior_err(zval < 3, :);
% pck_post_err = pck_post_err(zval < 3, :);
% pck_loo_err = pck_loo_err(zval < 3,:);
% weak_err = weak_err(zval < 3, :);

fprintf('Outliers count with zscore: %d\n', sum(zval > 3));

%%

val = pck_post_err(:,end);
Q1 = quantile(val,0.25);
Q3 = quantile(val,0.75);
IQR = Q3 - Q1;
out_mask = (val < (Q1 - 1.5 * IQR)) | (val > (Q3 + 1.5 * IQR));

% pck_prior_err = pck_prior_err(~out_mask, :);
% pck_post_err = pck_post_err(~out_mask, :);
% pck_loo_err = pck_loo_err(~out_mask,:);
% weak_err = weak_err(~out_mask, :);

fprintf('Outliers count with IQR: %d\n', sum(out_mask));

%% Deduce means with deviations

pck_prior_err_means = mean(pck_prior_err,1);
pck_post_err_means = mean(pck_post_err,1);
pck_loo_err_means = mean(pck_loo_err,1);
weak_err_means = mean(weak_err,1);

pck_prior_err_conf = 2*std(pck_prior_err,1);
pck_post_err_conf = 2*std(pck_post_err,1);
pck_loo_err_conf = 2*std(pck_loo_err,1);
weak_err_conf = 2*std(weak_err,1);

%% Select ranges and fit

% Select range
iters = 5:iterations;

weak_err_means = weak_err_means(iters);
weak_err_conf = weak_err_conf(iters);
pck_prior_err_means = pck_prior_err_means(iters);
pck_post_err_means = pck_post_err_means(iters);
pck_loo_err_means = pck_loo_err_means(iters);
pck_prior_err_conf = pck_prior_err_conf(iters);
pck_post_err_conf = pck_post_err_conf(iters);
pck_loo_err_conf = pck_loo_err_conf(iters);


% Fit errors
[a,~] = polyfit(iters, log10(pck_post_err_means), 1);
[b,~] = polyfit(iters, log10(pck_loo_err_means), 1);
[c,~] = polyfit(iters, log10(pck_prior_err_means), 1);

fprintf("Prior validation convergence rate: %f\n", -c(1))
fprintf("Posterior validation convergence rate: %f\n", -a(1))
fprintf("LOO convergence rate: %f\n", -b(1))

fprintf("Prior validation final error: %g\n", pck_prior_err_means(end))
fprintf("Posterior validation final error: %g\n", pck_post_err_means(end))
fprintf("LOO final error: %f\n", pck_loo_err_means(end))
    
%% Plots weak error


figure
hold on
errorbar(iters, weak_err_means, weak_err_conf, "LineStyle","none", "Color", "black")
scatter(iters, weak_err_means, 'filled')
hold off
set(gca, 'YScale', 'log')
xlabel('Iteration')
ylabel('Weak error $|\hat{Z} - \mathcal{Z}|$', 'Interpreter','latex')
ylim([min(weak_err_means),4e-2])
%title('Evidence convergence')


figure
hold on
plot(iters, 10.^(c(1) .* iters + c(2)), 'DisplayName', 'Prior error Fit', 'Color', "black", 'LineStyle', '-', 'LineWidth', 2)
plot(iters, 10.^(a(1) .* iters + a(2)), 'DisplayName', 'Post error Fit', 'Color', "black", 'LineStyle', ':', 'LineWidth', 2)
plot(iters, 10.^(b(1) .* iters + b(2)), 'DisplayName', 'LOO error Fit', 'Color',"black" , 'LineStyle', '--', 'LineWidth', 2)
errorbar(iters, pck_prior_err_means,pck_prior_err_conf, 'HandleVisibility','off', 'Color', "#000000", "LineStyle","none")
errorbar(iters, pck_post_err_means,pck_post_err_conf, 'HandleVisibility','off', 'Color', "#000000", "LineStyle","none")
errorbar(iters, pck_loo_err_means, pck_loo_err_conf, 'HandleVisibility','off', 'Color', "#000000", "LineStyle","none")
scatter(iters, pck_prior_err_means, 60, 'filled', 'DisplayName', 'Prior validation error', 'MarkerFaceColor', "#EDB120")
scatter(iters, pck_post_err_means, 60, 'filled', 'DisplayName', 'Posterior validation error', 'MarkerFaceColor', "#7E2F8E")
scatter(iters, pck_loo_err_means, 60, 'filled', 'DisplayName', 'PCK LOO error', 'MarkerFaceColor', "#77AC30")
hold off
set(gca, 'YScale', 'log')
xlabel('Iteration')
ylabel('Validation/LOO Error')
%title('Posterior or prior error vs. LOO error')
grid on
ylim([min(pck_post_err_means) / 5, 1000])
legend('interpreter','latex', 'FontSize', 10, 'NumColumns', 2)




