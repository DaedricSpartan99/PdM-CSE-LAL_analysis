function cpck = clustered_PCK(Opts)

    %% Input options
    %
    % Opts.MetaOpts             Options for the local metamodel
    % Opts.ExpDesign.X          Experimental design X
    % Opts.ExpDesign.Y          Experimental design Y
    % Opts.DBEpsilon            DBScan sphere radius (optional)
    % Opts.DBMinPts             DBScan minimum point

    %% Output options
    %
    % CPCK.norm                 Informations for normalization of ED
    % CPCK.models               (cell) Actual trained models
    % CPCK.labels               (array) DBScan labels in ED

    %% Perform DBScan clustering

    scanspace = [Opts.ExpDesign.X, Opts.ExpDesign.Y];

    cpck.norm.mean = mean(scanspace);
    cpck.norm.std = std(scanspace);

    cpck.norm.Z = (scanspace - cpck.norm.mean) ./ cpck.norm.std;
    cpck.dimX = size(Opts.ExpDesign.X,2);
    cpck.dimY = size(Opts.ExpDesign.Y,2);

    cpck.norm.meanX = cpck.norm.mean(:,1:cpck.dimX);
    cpck.norm.stdX = cpck.norm.std(:,1:cpck.dimX);
    
    % Find DBScan radius
    if ~isfield(Opts, 'DBEpsilon')

        dbscan_kD = pdist2(cpck.norm.Z, cpck.norm.Z,'euc','Smallest',Opts.DBMinPts);
        dbscan_kD = sort(dbscan_kD(end,:));

        [~,eps_dbscan_ind] = knee_pt(dbscan_kD, 1:length(dbscan_kD));
        Opts.DBEpsilon = dbscan_kD(eps_dbscan_ind);
    end

    nb_labels = 0;

    % Finalize with an actual scan
    while nb_labels == 0
        dbscan_labels = dbscan(cpck.norm.Z, Opts.DBEpsilon, Opts.DBMinPts);

        unique_labels = sort(unique(dbscan_labels));
        
        if any(unique_labels == -1)
            unique_labels = unique_labels(2:end); % Skip outliers
        end

        nb_labels = length(unique_labels);

        Opts.DBEpsilon = Opts.DBEpsilon * 1.2;

        fprintf("CPCK: trying epsilon, %f\n", Opts.DBEpsilon)
    end

    Opts.DBEpsilon = Opts.DBEpsilon / 1.2;

    fprintf("CPCK: Clusters found: %d\n", nb_labels)


    %% Include outliers in ED by distance minimization (similar to classification)

    outZ = cpck.norm.Z(dbscan_labels == -1);
    distances = pdist2(outZ, cpck.norm.Z(dbscan_labels ~= -1),'euc');

    [~, idx] = min(distances, [], 2);
    labels_no_out = dbscan_labels(dbscan_labels ~= -1);
    out_best_labels = labels_no_out(idx);

    cpck.labels = dbscan_labels;
    cpck.labels(dbscan_labels == -1) = out_best_labels;

    
    %% Train models
    
    cpck.models = cell(nb_labels,1);

    for i = 1:nb_labels

        MetaOpts = Opts.MetaOpts;
        MetaOpts.ExpDesign.X = Opts.ExpDesign.X(cpck.labels == i,:);
        MetaOpts.ExpDesign.Y = Opts.ExpDesign.Y(cpck.labels == i,:);

        cpck.models{i} = uq_createModel(MetaOpts, '-private');
    end

    cpck.Options = Opts;

    %% Setup UQ Metamodel
    MetaModelOpts.mFile = 'clustered_PCK_eval';
    MetaModelOpts.Parameters = cpck;
    MetaModelOpts.isVectorized = true;

    cpck.MetaModel = uq_createModel(MetaModelOpts, '-private');
end
