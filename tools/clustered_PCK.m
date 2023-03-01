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

    cpck.norm.meanX = mean(hyperspace);
    cpck.norm.stdX = std(hyperspace);
    cpck.norm.Z = (Opts.ExpDesign.X - cpkc.norm.meanX) / cpck.norm.stdX;
    
    % Find DBScan radius
    if ~isfield(Opts, 'DBEpsilon')

        dbscan_kD = pdist2(cpck.norm.Z, cpck.norm.Z,'euc','Smallest',Opts.MinPts);
        dbscan_kD = sort(dbscan_kD(end,:));

        [~,eps_dbscan_ind] = knee_pt(dbscan_kD, 1:length(dbscan_kD));
        Opts.DBEpsilon = kD(eps_dbscan_ind);
    end

    % Finalize with an actual scan
    dbscan_labels = dbscan(cpck.norm.Z, Opts.DBEpsilon, Opts.DBMinPts);

    unique_labels = unique(dbscan_labels);
    unique_labels = unique_labels(2:end); % Skip outliers
    nb_labels = length(unique_labels);

    fprintf("CPCK: Clusters found: %d", nb_labels)


    %% Include outliers in ED by distance minimization (similar to classification)

    outZ = cpck.norm.Z(dbscan_labels == -1);
    distances = pdist2(outZ, cpck.norm.Z(dbscan_labels ~= -1),'euc','Smallest',Opts.MinPts);

    [~, idx] = min(distances, [], 2);
    out_best_labels = CPCK.labels(idx);

    cpck.labels = out_best_labels;

    
    %% Train models
    
    cpck.models = cell(nb_labels,1);

    for i = 1:nb_labels

        MetaOpts = Opts.MetaOpts;
        MetaOpts.ExpDesign.X = cpck.norm.Z(cpck.labels == i,:);
        MetaOpts.ExpDesign.Y = Opts.ExpDesign.Y(cpck.labels == i,:);

        cpck.models{i} = uq_createModel(MetaOpts, '-private');
    end
end
