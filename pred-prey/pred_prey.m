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

PriorOpts.Marginals(2).Name = ('beta');
PriorOpts.Marginals(2).Type = 'LogNormal';
PriorOpts.Marginals(2).Moments = [0.05 0.005];

PriorOpts.Marginals(3).Name = ('gamma');
PriorOpts.Marginals(3).Type = 'LogNormal';
PriorOpts.Marginals(3).Moments = [1 0.1];

PriorOpts.Marginals(4).Name = ('delta');
PriorOpts.Marginals(4).Type = 'LogNormal';
PriorOpts.Marginals(4).Moments = [0.05 0.005];

PriorOpts.Marginals(5).Name = ('initH');
PriorOpts.Marginals(5).Type = 'LogNormal';
PriorOpts.Marginals(5).Parameters = [log(10) 1];

PriorOpts.Marginals(6).Name = ('initL');
PriorOpts.Marginals(6).Type = 'LogNormal';
PriorOpts.Marginals(6).Parameters = [log(10) 1];

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
LOpts.Parameters.DiscrepancyModel = DiscrepancyModel;
LOpts.Parameters.DiscrepancyDim = 2; % length of DiscrepancyOpts
LOpts.Parameters.myData = myData;  % Vectorize measurements

LogLikelihoodModel = uq_createModel(LOpts);

%% Bayesian analysis

LALOpts.Bus.logC = -10.; % best value: 1 / (max L + small_quantity) 
LALOpts.Bus.p0 = 0.1;                            % Quantile probability for Subset
LALOpts.Bus.BatchSize = 1e4;                             % Number of samples for Subset simulation
LALOpts.Bus.MaxSampleSize = 1e5;
LALOpts.MaximumEvaluations = 50;
LALOpts.ExpDesign.FilterZeros = false;
LALOpts.ExpDesign.InitEval = 5;
LALOpts.PlotLogLikelihood = true;

LALOpts.PCE.MinDegree = 4;
LALOpts.PCE.MaxDegree = 32;

LALOpts.LogLikelihood = LogLikelihoodModel;
LALOpts.Prior = myPriorDist;
LALOpts.Discrepancy = DiscrepancyOpts;

LALAnalysis = lal_analysis(LALOpts);