# LAL_analysis
Likelihood active learning allows to limit the number of evaluations necessary for a Bayesian analysis.

## Requires

* UQLab
* Optimization Toolbox
* Global Optimization Toolbox

## Basic Options

| Field                         |  Functionality                              | Type        |
|-------------------------------|:-------------------------------------------:|-------------|
| Opts.MaximumEvaluations       |  Number of LAL iterations                   | int > 0     |
| Opts.ExpDesign.X              |  Initial experimental design X              | array N x M |
| Opts.ExpDesign.LogLikelihood  |  Initial experimental design log-likelihood | array N x 1 |
| Opts.ExpDesign.InitEval(*)    |  Alternative to ExpDesign.LogLikelihood     | int > 0     |
| Opts.LogLikelihood            |  Log-likelihood forward model               | handle      | 
| Opts.Prior                    |  Prior input                                | UQInput     |
    
## Validation options

| Field                         |  Functionality                              | Type        |
|-------------------------------|:-------------------------------------------:|-------------|
| Opts.PlotLogLikelihood        |  Plot algorithm progress                    | logical     |
| Opts.Validation.PostSamples   |  Validation set X for PCK                   | array * x M |
| Opts.Validation.LogLikelihood |  Validation set Y for PCK                   | array * x 1 |
    
    
## BuS, PCK and ED options

| Field                         |  Functionality                              | Type        |
|-------------------------------|:-------------------------------------------:|-------------|
| Opts.Bus.logC                 | Specify constant c                          | double      |
| Opts.Bus.p0                   | SuS quantile probability                    | double [0, 0.5] |
| Opts.Bus.BatchSize            | Samples per subset in SuS                   | int > 0   |
| Opts.Bus.MaxSampleSize        | Max samples in SuS                          | int > 0   |
| Opts.Bus.CStrategy            | Define how to determine constant c          | 'max', 'maxpck', 'latest' or 'delaunay' |
| Opts.MetaOpts                 | Options for surrogate                       | Struct   |
| Opts.cleanQuantile            | ED ignore below log-likelihood quantile     | double [0, 1] |
| Opts.StoreBusResults          | Take in memory results                      | logical |
| Opts.ClusteredMetaModel       | Use a clustered metamodel (experimental)    | logical |

## Samples's clustering options
   
| Field                         |  Functionality                              | Type        |
|-------------------------------|:-------------------------------------------:|-------------|
| Opts.ClusterRange             | Number of parallel calls (clusters)         |   int |
| Opts.OptMode                  | Set learning function                       | 'single' or 'clustering' |

## Output fields

| Field                                 |  Functionality                                 | Type        |
|---------------------------------------|:----------------------------------------------:|-------------|
| LALAnalysis.ExpDesign.X               | Enriched experimental design                   | Array N_out x M |
| LALAnalysis.ExpDesign.LogLikelihood   | Enriched experimental design                   | Array N_out x 1 |
| LALAnalysis.ExpDesign.UnNormPosterior | Enriched design of posterior                   | Array N_out x 1 |
| LALAnalysis.BusAnalysis (*)           | BusAnalysis intermediate results               | Struct |
| LALAnalysis.Evidence                  | Bayesian evidence Z (normalization constant)   | double > 0 |          
| LALAnalysis.Opts                      | LALAnalysis options struct                     | Struct |

# Examples

## Simple setup

<pre><code>
% Basic options
LALOpts.MaximumEvaluations = 30;
LALOpts.ExpDesign.X = myX;
LALOpts.ExpDesign.LogLikelihood = mylogL;
LALOpts.LogLikelihood = myLogLikelihood;
LALOpts.Prior = myFullPrior;

% Determination of c
LALOpts.Bus.CStrategy = 'maxpck';
LALOpts.Maxpck.alpha = 0.05;

% SuS samples clustering options
LALOpts.ClusterRange = 1;

% PC-Kriging options
LALOpts.MetaOpts.MetaType = 'PCK';
LALOpts.MetaOpts.PCE.Degree = 0:4;
LALOpts.MetaOpts.Mode = 'optimal';   
LALOpts.MetaOpts.Kriging.Optim.Bounds = [0.001; 1000];

% Experimental design reduction for training
LALOpts.cleanQuantile = 0.025;

% Bus options
LALOpts.Bus.BatchSize = 5000;
LALOpts.Bus.MaxSampleSize = 500000;

% Validation options
LALOpts.Validation.PostSamples = post_samples;
LALOpts.Validation.PostLogLikelihood = post_logL_samples;
LALOpts.Validation.PriorSamples = prior_samples;
LALOpts.Validation.PriorLogLikelihood = prior_logL_samples;

% Activate storage and plot progress
LALOpts.PlotLogLikelihood = true;
LALOpts.StoreBusResults = true;

LALAnalysis = lal_analysis(LALOpts);
</code></pre>

## Obtain handles

<pre><code>
% Setup Bayesian Analysis
BayesOpts.Solver.Type = 'None';
BayesOpts.ForwardModel = myModel;
handlesBayesAnalysis = uq_createAnalysis(BayesOpts);

% Get handles
myLogLikelihood = handlesBayesAnalysis.LogLikelihood;
myInput = handlesBayesAnalysis.Internal.FullPrior;
</code></pre>

## Learn more about the math behind

* [https://drive.google.com/file/d/1UQFo-dxXswSPfIqeBDFJwV2JcDodUfu3/view?usp=sharing](Thesis presentation)
* [https://drive.google.com/file/d/1jHn4E5WANII1FbTdkE6nLmnVLnIaEWdA/view?usp=sharing](Thesis poster)

## Commissioners and credits
* [https://www.geomod.ch/](Geomod SA, Lausanne)
* [https://baug.ethz.ch/en/department/people/staff/personen-detail.MTQ3NDU3.TGlzdC82NzksLTU1NTc1NDEwMQ==.html](Thesis supervisor, ETH, Dr. Stefano Marelli)
* [https://baug.ethz.ch/en/department/people/staff/personen-detail.MTg3NzUx.TGlzdC82NzksLTU1NTc1NDEwMQ==.html](Thesis commissioner, ETH, Prof. Bruno Sudret)
* [https://people.epfl.ch/marco.picasso](Thesis commissioner, EPFL, Prof. Marco Picasso)
