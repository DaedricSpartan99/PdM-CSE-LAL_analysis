clearvars
rng(100,'twister')
uqlab

addpath('../lal')
addpath('../tools')

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
%PriorOpts.Marginals(5).Type = 'Constant';
%PriorOpts.Marginals(5).Parameters = 1.2317e+04;           % (N/m)
PriorOpts.Marginals(5).Type = 'Gaussian';
PriorOpts.Marginals(5).Moments = [12000 600]; % (N/m)

myPriorDist = uq_createInput(PriorOpts);

%% Measurement setup

myData.y = [12.84; 13.12; 12.13; 12.19; 12.67]/1000; % (m)
myData.Name = 'Mid-span deflection';

DiscrepancyOpts(1).Type = 'Gaussian';
DiscrepancyOpts(1).Parameters = 1.2448e-06;%var(myData.y);
% Discrepancy standard deviation of the variance: 1.7026e-06

%% Bayesian invertion

BayesOpts.Type = 'Inversion';
BayesOpts.Data = myData;
BayesOpts.Discrepancy = DiscrepancyOpts;

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

%% Prior and post predictive samples

% prior predictive
prior_pred_samples = uq_evalModel( myForwardModel, [repmat([0.15, 0.3, 5],size(prior_samples,1),1),prior_samples]);

% posterior_predictive
post_pred_samples = uq_evalModel(myForwardModel, [repmat([0.15, 0.3, 5],size(post_samples,1),1),post_samples]);

%% Prior Histograms

figure
histogram(prior_samples(:,1), 30, 'FaceColor', 'green');
ylabel('E')
yticks([])
xlim([min(prior_samples(:,1)), max(prior_samples(:,1))])
exportgraphics(gcf,'images/prior_E.png','BackgroundColor','none','Resolution',300)
%saveas(gcf,'images/prior_E','epsc')

figure
histogram(prior_samples(:,2), 30, 'FaceColor', 'cyan');
ylabel('P')
yticks([])
xlim([min(prior_samples(:,2)), max(prior_samples(:,2))])
exportgraphics(gcf,'images/prior_P.png','BackgroundColor','none','Resolution',300)

figure
histogram(prior_pred_samples, 30, 'FaceColor', "#EDB120")
xlabel('V')
yticks([])
xlim([min(prior_pred_samples), max(prior_pred_samples)])
exportgraphics(gcf,'images/prior_V.png','BackgroundColor','none','Resolution',300)

%% Post Histograms

figure
histogram(post_samples(:,1), 30, 'FaceColor', "#77AC30");
ylabel('E')
yticks([])
xlim([min(prior_samples(:,1)), max(prior_samples(:,1))])
exportgraphics(gcf,'images/post_E.png','BackgroundColor','none','Resolution',300)

figure
histogram(post_samples(:,2), 30, 'FaceColor', 'blue');
ylabel('P')
yticks([])
xlim([min(prior_samples(:,2)), max(prior_samples(:,2))])
exportgraphics(gcf,'images/post_P.png','BackgroundColor','none','Resolution',300)

figure
histogram(post_pred_samples, 30, 'FaceColor', 	"#D95319")
xlabel('V')
yticks([])
xlim([min(prior_pred_samples), max(prior_pred_samples)])
exportgraphics(gcf,'images/post_V.png','BackgroundColor','none','Resolution',300)

