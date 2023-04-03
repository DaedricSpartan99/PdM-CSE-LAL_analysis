% likelihood active learning

clearvars
rng(100,'twister')
uqlab

%% Bayesian inversion definition (single measurement)

% Measurement (single)
Y = 8.                 % measurement value
discrepancy = 0.01^2   % measurement variance

% Prior definition
PriorOpts.Name = 'Prior'
PriorOpts.Marginals(1).Type = 'Gaussian'    % prior form
PriorOpts.Marginals(1).Moments = [0., 1.]   % prior mean and variance
PriorInput = uq_createInput(PriorOpts);

% Model definition
a = 2
ModelOpts.Name = 'myModel';
ModelOpts.mFile = 'myModel'           % model selection
ModelOpts.Parameters.a = a;
myModel = uq_createModel(ModelOpts);  


% Hyperparameters
c = sqrt(2*pi*discrepancy) * 1.2    % best value: 1 / (max L + small_quantity) 
Pt = 0.1                            % Quantile probability for Subset
K = 1e5                             % Number of samples for Subset simulation
N_max = 20;

%% Likelihood definition for inversion
%LOpts.Name = 'log_likelihood_model';
LOpts.mFile = 'log_likelihood_model';
LOpts.Parameters.Model = myModel;
LOpts.Parameters.Measurement = Y;
LOpts.Parameters.Discrepancy = discrepancy;     % discrepancy model

LogLikelihoodModel = uq_createModel(LOpts);

%% Initial experimental design

X0 = uq_getSample(10);
logL0 = uq_evalModel(LogLikelihoodModel, X0);
%logL_threshold = 1e-14;

%% Active Learning loop

% Input: Experimental design setup
X = X0;
logL = logL0;

% Output: Enriched X, logL with points of interest




% BuS input definition
% Setup P random variable
BusPriorOpts.Name = 'BusPrior';
BusPriorOpts.Marginals(1).Name = 'P';
BusPriorOpts.Marginals(1).Type = 'Uniform';
BusPriorOpts.Marginals(1).Parameters = [0, 1];

% Setup X random variable
BusPriorOpts.Marginals(2).Type = PriorInput.Marginals(1).Type;
BusPriorOpts.Marginals(2).Parameters = PriorInput.Marginals(1).Parameters;

% Begin iterations
for i = 1:N_max

    % Construct a PC-Kriging surrogate of the log-likelihood
    PCKOpts.Type = 'Metamodel';
    PCKOpts.MetaType = 'PCK';
    PCKOpts.Mode = 'sequential';
    PCKOpts.FullModel = LogLikelihoodModel;
    PCKOpts.PCE.Degree = 2:2:12;
    PCKOpts.PCE.Method = 'LARS';
    PCKOpts.ExpDesign.X = X;
    PCKOpts.ExpDesign.Y = logL;
    PCKOpts.Kriging.Corr.Family = 'Gaussian';

    logL_PCK = uq_createModel(PCKOpts, '-private');
    
    % TODO: Determine optimal c = 1 / max(L)

    % Construct BuS Limit state function
    LSFOpts.mFile = 'limit_state_model';
    LSFOpts.isVectorized = true;
    LSFOpts.Parameters.c = c;
    LSFOpts.Parameters.LogLikelihood = logL_PCK;   % Set the surrogate model


    % Setup and run the Subset simulation
    SSOpts.Type = 'Reliability';
    SSOpts.Method = 'Subset';
    SSOpts.SubsetSim.p0 = Pt;
    SSOpts.Simulation.BatchSize = K;
    SSOpts.Model = uq_createModel(LSFOpts, '-private');
    SSOpts.Input = uq_createInput(BusPriorOpts, '-private');
    SSimAnalysis = uq_createAnalysis(SSOpts, '-private');

    % get samples of 
    post_samples_PX = SSimAnalysis.Results.History.X{end};

    % evaluate U-function on the limit state function
    % Idea: maximize misclassification probability
    [mean_post_LSF, var_post_LSF] = uq_evalModel(SSOpts.Model, post_samples_PX);
    cost_LSF = abs(mean_post_LSF) ./ sqrt(var_post_LSF);
    [~, opt_index] = min(cost_LSF);
    xopt = post_samples_PX(opt_index,2);

    % Add to experimental design
    X = [X; xopt];
    logL = [logL; uq_evalModel(LogLikelihoodModel, xopt) ];
end


% END of the algorithm



%% Likelihood visualization

% Plot Likelihood experimental design
figure
scatter(X, exp(logL))
xlabel('$x$')
ylabel('$L$')
title('Likelihood approximation experimental design')

%% Posterior samples

N_samples = 2000;

% Run a subset simulation
% Construct a PC-Kriging surrogate of the log-likelihood
PCKOpts.Type = 'Metamodel';
PCKOpts.MetaType = 'PCK';
PCKOpts.Mode = 'sequential';
PCKOpts.FullModel = LogLikelihoodModel;
PCKOpts.PCE.Degree = 2:2:12;
PCKOpts.PCE.Method = 'LARS';
PCKOpts.ExpDesign.X = X;
PCKOpts.ExpDesign.Y = logL;
PCKOpts.Kriging.Corr.Family = 'Gaussian';

logL_PCK = uq_createModel(PCKOpts, '-private');

% Construct BuS Limit state function
LSFOpts.mFile = 'limit_state_model';
LSFOpts.isVectorized = true;
LSFOpts.Parameters.c = c;
LSFOpts.Parameters.LogLikelihood = logL_PCK;   % Set the surrogate model


% Setup and run the Subset simulation
SSOpts.Type = 'Reliability';
SSOpts.Method = 'Subset';
SSOpts.SubsetSim.p0 = Pt;
SSOpts.Simulation.BatchSize = N_samples;
SSOpts.Model = uq_createModel(LSFOpts, '-private');
SSOpts.Input = uq_createInput(BusPriorOpts, '-private');
SSimAnalysis = uq_createAnalysis(SSOpts, '-private');

% get samples of 
post_samples_X = SSimAnalysis.Results.History.X{end}(:,2);

uq_selectInput('Prior');
prior_samples_X = uq_getSample(N_samples);

figure
hold on
histogram(prior_samples_X)
histogram(post_samples_X)
hold off
xlabel('$x')
ylabel('Occurrences')
title('Prior vs Posterior sample comparison')
legend('Prior', 'Posterior')