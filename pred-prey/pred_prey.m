clearvars
rng(100,'twister')
uqlab

addpath('../lal')

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
%PriorOpts.Marginals(1).Parameters = [1];

PriorOpts.Marginals(2).Name = ('beta');
PriorOpts.Marginals(2).Type = 'LogNormal';
PriorOpts.Marginals(2).Moments = [0.05 0.005];
%PriorOpts.Marginals(2).Type = 'Constant';
%PriorOpts.Marginals(2).Parameters = [0.05];

PriorOpts.Marginals(3).Name = ('gamma');
PriorOpts.Marginals(3).Type = 'LogNormal';
PriorOpts.Marginals(3).Moments = [1 0.1];
%PriorOpts.Marginals(3).Type = 'Constant';
%PriorOpts.Marginals(3).Parameters = [1];

PriorOpts.Marginals(4).Name = ('delta');
PriorOpts.Marginals(4).Type = 'LogNormal';
PriorOpts.Marginals(4).Moments = [0.05 0.005];
%PriorOpts.Marginals(4).Type = 'Constant';
%PriorOpts.Marginals(4).Parameters = [0.05];

PriorOpts.Marginals(5).Name = ('initH');
PriorOpts.Marginals(5).Type = 'LogNormal';
PriorOpts.Marginals(5).Parameters = [log(10) 1];

PriorOpts.Marginals(6).Name = ('initL');
PriorOpts.Marginals(6).Type = 'LogNormal';
PriorOpts.Marginals(6).Parameters = [log(10) 1];
%PriorOpts.Marginals(6).Type = 'Constant';
%PriorOpts.Marginals(6).Parameters = [0.5];

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
%SigmaOpts.Marginals(1).Type = 'Constant';
%SigmaOpts.Marginals(1).Parameters = [exp(-1. + 1. / 2)];
SigmaOpts.Marginals(1).Type = 'Lognormal';
SigmaOpts.Marginals(1).Parameters = [-1 1];

SigmaDist1 = uq_createInput(SigmaOpts);

SigmaOpts.Marginals(1).Name = 'Sigma2H';
%SigmaOpts.Marginals(1).Type = 'Constant';
%SigmaOpts.Marginals(1).Parameters = [exp(-1. + 1. / 2)];
SigmaOpts.Marginals(1).Type = 'Lognormal';
SigmaOpts.Marginals(1).Parameters = [-1 1];

SigmaDist2 = uq_createInput(SigmaOpts);

DiscrepancyOpts(1).Type = 'Gaussian';
DiscrepancyOpts(1).Prior = SigmaDist1;
DiscrepancyOpts(2).Type = 'Gaussian';
DiscrepancyOpts(2).Prior = SigmaDist2;

% Create discrepancy model
DiscModelOpts.Name = 'discrepancy_model';
DiscModelOpts.mFile = 'discrepancy_model';
DiscrepancyModel = uq_createModel(DiscModelOpts);

%% Log-likelihood definition

LOpts.Name = 'log_likelihood_model';
LOpts.mFile = 'log_likelihood_model';
LOpts.Parameters.ForwardModel = myForwardModel;
LOpts.Parameters.Amplification = 1e-4;
LOpts.Parameters.DiscrepancyModel = DiscrepancyModel;
LOpts.Parameters.DiscrepancyDim = 2; % length of DiscrepancyOpts
LOpts.Parameters.myData = myData;  % Vectorize measurements

LogLikelihoodModel = uq_createModel(LOpts);

%% Bayesian analysis

%LALOpts.Bus.logC = 1.5e4; % best value: -max log(L) 
LALOpts.Bus.p0 = 0.1;                            % Quantile probability for Subset
LALOpts.Bus.BatchSize = 1e4;                             % Number of samples for Subset simulation
LALOpts.Bus.MaxSampleSize = 1e5;
LALOpts.MaximumEvaluations = 200;
LALOpts.ExpDesign.FilterZeros = false;
LALOpts.ExpDesign.InitEval = 10;
LALOpts.PlotLogLikelihood = true;
LALOpts.PlotTarget = 5;
LALOpts.CStrategy = 'max';

LALOpts.PCE.MinDegree = 2;
LALOpts.PCE.MaxDegree = 16;

LALOpts.LogLikelihood = LogLikelihoodModel;
LALOpts.Prior = myPriorDist;
LALOpts.Discrepancy = DiscrepancyOpts;

LALAnalysis = lal_analysis(LALOpts);

%% Analysis: plot of experimental design and real log-likelihood on marginal 5 and 6

% generate a full set corresponding
n_samples = 100;
X5_samples = linspace(1, max(LALAnalysis.ExpDesign.X(:,5)) , n_samples);
X6_samples = linspace(1, max(LALAnalysis.ExpDesign.X(:,6)) , n_samples);
%X5_samples = linspace(15, 25, n_samples);
%X6_samples = linspace(8, 16 , n_samples);

[X5_mesh,X6_mesh] = meshgrid(X5_samples,X6_samples);
X56 = [X5_mesh(:) X6_mesh(:)];

X_samples = [repmat([1., 0.05, 1., 0.05], size(X56,1), 1), X56, repmat([exp(-1 + 1. / 2)], size(X56,1), 2)];

logL_real = uq_evalModel(LogLikelihoodModel, X_samples) ./ LOpts.Parameters.Amplification;

% filter results
logL_real(logL_real < -1e5) = -1e5;

figure
xlabel('X_5')
ylabel('X_6')
zlabel('logL')
hold on
scatter3(LALAnalysis.ExpDesign.X(:,5), LALAnalysis.ExpDesign.X(:,6), LALAnalysis.ExpDesign.LogLikelihood ./ LOpts.Parameters.Amplification , "filled")
surfplot = surf(X5_samples, X6_samples, reshape(logL_real, n_samples, n_samples));
surfplot.EdgeColor = 'none';
hold off
legend('Experimental design', 'Real Log-Likelihood')

