clearvars
rng(100,'twister')
uqlab

%% Bayesian inversion definition (single measurement)

% Measurement (single)
Y = 1.                 % measurement value
discrepancy = 0.05^2   % measurement variance

% Prior definition
prior_type = 'Gaussian'    % prior form
prior_moments = [0., 1.]   % prior mean and variance

% Model definition
a = 2
ModelOpts.mFile = 'myModel'           % model selection
ModelOpts.Parameters.a = a;
myModel = uq_createModel(ModelOpts);  

% Hyperparameters
c = sqrt(2*pi*discrepancy) * 1.1    % best value: 1 / (max L + small_quantity) 
Pt = 0.1                            % Quantile probability for Subset
K = 2e4                             % Number of samples for Subset simulation

%% Likelihood definition for inversion
LOpts.mFile = 'likelihood_model';
LOpts.Parameters.Model = myModel;
LOpts.Parameters.Measurement = Y;
LOpts.Parameters.Discrepancy = discrepancy;     % discrepancy model

LikelihoodModel = uq_createModel(LOpts);

%% Basic setup for adaptive learning reliability
ALROpts.Type = 'Reliability';
ALROpts.Method = 'ALR';
ALROpts.ALR.Reliability = 'Subset';
ALROpts.Display = 'verbose';

ALROpts.ALR.MaxAddedED = 20; % Maximum number of iterations

%% Limit state function definition
LSFOpts.mFile = 'limit_state_model';
LSFOpts.isVectorized = true;
LSFOpts.Parameters.c = c;
LSFOpts.Parameters.epsilon = 1e-14;         % do not let L fall to zero, address singularities
LSFOpts.Parameters.Likelihood = LikelihoodModel;
ALROpts.Model = uq_createModel(LSFOpts);

%% Prior composed input definition

% Setup X random variable
PriorOpts.Marginals(1).Name = 'X';
PriorOpts.Marginals(1).Type = prior_type;
PriorOpts.Marginals(1).Moments = prior_moments;

% Setup P random variable
PriorOpts.Marginals(2).Name = 'P';
PriorOpts.Marginals(2).Type = 'Uniform';
PriorOpts.Marginals(2).Parameters = [0, 1];

ALROpts.Input = uq_createInput(PriorOpts);

%% Subset simulation parameters

% Quantile probability
ALROpts.Subset.p0 = Pt;

% Number of samples
ALROpts.ALR.Simulation.BatchSize = K;
ALROpts.ALR.Simulation.MaxSampleSize = 20 * K;
ALROpts.ALR.LearningFunction = 'U';

%% Initial experimental design

%N_exp = 4;

%XP = uq_getSample(ALROpts.Input, N_exp);
%L_exp = uq_evalModel(LikelihoodModel, XP(:,2));
%P = min(c * L_exp, 1.);

% assign initial experimental design
%XP(:,1) = P;
%G = log(P) - log(c) - log(L_exp + LSFOpts.Parameters.epsilon );

%ALROpts.ALR.IExpDesign.X = XP;
%ALROpts.ALR.IExpDesign.G = G;

%% PC-K setup

ARLOpts.ALR.Metamodel = 'PCK';
PCKOpts.PCE.Degree = 2:2:16;

%PCKOpts.PCE.LARS.HybridLoo = 0;
%PCKOpts.PCE.Method = 'OLS';
%PCKOpts.PCE.OLS.ModifiedLOO = 0;

ARLOpts.ALR.PCK = PCKOpts;

%% Execute active learning analysis
ALRAnalysis = uq_createAnalysis(ALROpts);

%% Get samples results
samples = ALRAnalysis.Results.History.X;

%% Display
uq_display(ALRAnalysis)

%% Verify Pf
post_variance = (a^2 + discrepancy);
ALRAnalysis.Results

Z_ana = exp(-Y^2/post_variance/2) / sqrt(2*pi*post_variance)
Z_estimated = ALRAnalysis.Results.Pf / c

%% Plot PCK
