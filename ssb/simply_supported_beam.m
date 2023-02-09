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
%PriorOpts.Marginals(5).Type = 'Constant';
%PriorOpts.Marginals(5).Parameters = 1.2317e+04;           % (N/m)
PriorOpts.Marginals(5).Type = 'Gaussian';
PriorOpts.Marginals(5).Moments = [12000 600]; % (N/m)

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

%% Bayesian analysis (tuning first peaks step)

%LALOpts.Bus.logC = 0; %-max(post_logL_samples); % best value: -max log(L) 
%LALOpts.Bus.p0 = 0.1;                            % Quantile probability for Subset
%LALOpts.Bus.BatchSize = 1e3;                             % Number of samples for Subset simulation
%LALOpts.Bus.MaxSampleSize = 1e4;
LALOpts.MaximumEvaluations = 5;
LALOpts.ExpDesign.InitEval = 15;
LALOpts.PlotLogLikelihood = true;
LALOpts.Bus.CStrategy = 'max';

%LALOpts.PCK.PCE.Degree = 0:5;
LALOpts.PCK.PCE.Method = 'LARS';
%LALOpts.PCK.PCE.PolyTypes = {'Hermite', 'Hermite'};
%LALOpts.PCK.Optim.Method = 'CMAES';
%LALOpts.PCK.Kriging.Optim.MaxIter = 1000;
%LALOpts.PCK.Kriging.Corr.Family = 'Gaussian';
%LALOpts.PCK.Kriging.Corr.Family = 'Matern-3_2';
%LALOpts.PCK.Kriging.Corr.Type = 'Separable';
%LALOpts.PCK.Kriging.Corr.Type = 'ellipsoidal';
%LALOpts.PCK.Kriging.theta = 9.999;
%LALOpts.PCK.Display = 'verbose';

LALOpts.LogLikelihood = refBayesAnalysis.LogLikelihood;
LALOpts.Prior = refBayesAnalysis.Internal.FullPrior;

LALOpts.Validation.PostSamples = post_samples;
LALOpts.Validation.PostLogLikelihood = post_logL_samples;
LALOpts.Validation.PriorSamples = prior_samples;
LALOpts.Validation.PriorLogLikelihood = prior_logL_samples;

FirstLALAnalysis = lal_analysis(LALOpts);

%% Switch bayesian analysis (explorative and refinement steps)

refine_steps = 15;
explore_steps = 10;
max_switches = 1;

LALOpts.PCK.PCE.Degree = 1:7;
LALOpts.PCK.PCE.Method = 'LARS';

LALOpts.Bus.BatchSize = 5000;
LALOpts.Bus.MaxSampleSize = 500000;

%LALOpts.PCK.Kriging.Corr.Type = 'Separable';

LALOpts.LogLikelihood = refBayesAnalysis.LogLikelihood;
LALOpts.Prior = refBayesAnalysis.Internal.FullPrior;

LALOpts.cleanQuantile = 0.1;

LALOpts.Validation.PostSamples = post_samples;
LALOpts.Validation.PostLogLikelihood = post_logL_samples;
LALOpts.Validation.PriorSamples = prior_samples;
LALOpts.Validation.PriorLogLikelihood = prior_logL_samples;

exp_design = FirstLALAnalysis.ExpDesign;

for sw = 1:max_switches

    LALOpts.MaximumEvaluations = explore_steps;
    LALOpts.Bus.CStrategy = 'delaunay';
    LALOpts.ExpDesign = exp_design;
    LALOpts.PlotLogLikelihood = false;

    ExploreLALAnalysis = lal_analysis(LALOpts);
    
    LALOpts.MaximumEvaluations = refine_steps;
    LALOpts.ExpDesign = ExploreLALAnalysis.ExpDesign;
    LALOpts.Bus.CStrategy = 'maxpck';
    LALOpts.PlotLogLikelihood = true;

    RefineLALAnalysis = lal_analysis(LALOpts);
    exp_design = RefineLALAnalysis.ExpDesign;
end

LALAnalysis = RefineLALAnalysis;

%pck = LALAnalysis.BusAnalysis(end).Opts.LogLikelihood;

%% Analyisis

X = LALAnalysis.ExpDesign.X;
TT = [X; post_samples];
idx = [ones(size(X,1),1); zeros(post_samples_size,1)];

figure
gplotmatrix(TT,TT,idx)

%% Prior domain analysis

prior_samples = LALOpts.Validation.PriorSamples;

x1 = linspace(min(prior_samples(:,1)), max(prior_samples(:,1)), 100);
x2 = linspace(min(prior_samples(:,2)), max(prior_samples(:,2)), 100);

[x_grid_1, x_grid_2] = meshgrid(x1,x2);
x_grid = [x_grid_1(:), x_grid_2(:)];

logL_real = LALOpts.LogLikelihood(x_grid);
logL_pck = uq_evalModel(pck, x_grid);

figure
hold on
scatter3(LALAnalysis.ExpDesign.X(:,1), LALAnalysis.ExpDesign.X(:,2), LALAnalysis.ExpDesign.LogLikelihood,  "filled")
surfplot = surf(x1, x2, reshape(logL_real, 100, 100));
surfplot.EdgeColor = 'none';
spck = surf(x1,x2,reshape(logL_pck, 100, 100),'FaceAlpha',0.5);
hold off
title('Prior domain log-likelihood')
xlabel('E')
ylabel('discrepancy')
zlabel('logL')
xlim([min(x1), max(x1)])
ylim([min(x2), max(x2)])
legend('Experimental design', 'Real logL', 'Surrogate logL')


%% Posterior domain analysis

x1 = linspace(min(post_samples(:,1)), max(post_samples(:,1)), 100);
x2 = linspace(min(post_samples(:,2)), max(post_samples(:,2)), 100);

[x_grid_1, x_grid_2] = meshgrid(x1,x2);
x_grid = [x_grid_1(:), x_grid_2(:)];

logL_real = LALOpts.LogLikelihood(x_grid);
logL_pck = uq_evalModel(pck, x_grid);

figure
hold on
scatter3(LALAnalysis.ExpDesign.X(:,1), LALAnalysis.ExpDesign.X(:,2), LALAnalysis.ExpDesign.LogLikelihood,  "filled")
surfplot = surf(x1, x2, reshape(logL_real, 100, 100));
surfplot.EdgeColor = 'none';
spck = surf(x1,x2,reshape(logL_pck, 100, 100),'FaceAlpha',0.5);
hold off
title('Posterior domain log-likelihood')
xlabel('E')
ylabel('discrepancy')
zlabel('logL')
xlim([min(x1), max(x1)])
ylim([min(x2), max(x2)])
legend('Experimental design', 'Real logL', 'Surrogate logL')
