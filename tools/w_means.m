function [labels, centroids] = w_means(X, W, k_range, varargin)

    %% Cross validate correct k
    F_score = zeros(max(k_range),1);
    F_score(:) = NaN;
    overall_mean = sum(W .* X) / sum(W);

    for k_ind = 1:length(k_range)

        % perform weighted k-means
        k = k_range(k_ind);
        [labels, centroids] = kw_means(X, W, k, varargin{:});

        % take label counts
        [uniques, counts] = count_unique(labels);

        % Filter unused labels and singularities
        centroids = centroids(uniques,:);

        finite_mask = all(isfinite(centroids),2);
        centroids = centroids(finite_mask,:);

        % compute square euclidean distances from overall mean
        F_sqdist = sum((centroids - overall_mean).^2, 2);

        % compute score
        k_act = length(centroids);
        F_score(k_act) = max(F_score(k_act), sum(counts .* F_sqdist) / (k-1));
    end

    %% Filter unchanged F_score
    finite_mask = isfinite(F_score);
    clusters = 1:length(F_score);
    clusters = clusters(finite_mask);
    F_score = F_score(finite_mask);
    %k_range = k_range(finite_mask);

    if sum(finite_mask) == 0
        disp("Error")
    end

    %% Find F_score curve elbow and apply actual weighted mean clustering
    if length(F_score) >= 3
        [~, opt_k_ind] = knee_pt(F_score, clusters);
        opt_k = clusters(opt_k_ind);
    else
        opt_k = clusters(end);
    end

    %% Finalize safely
    [labels, centroids] = kw_means(X, W, opt_k, varargin{:});

    [uniques, ~] = count_unique(labels);

    centroids = centroids(uniques,:);

    if size(centroids,1) ~= length(unique(labels))
        disp("Inconsistent size")
    end

    if any(any(~isfinite(centroids),2))
        disp("Infinite centroid")
    end

    finite_mask = all(isfinite(centroids),2);
    centroids = centroids(finite_mask,:);
    labels = labels(any(labels == uniques(finite_mask)', 2));

    if size(centroids,1) ~= length(unique(labels))
        disp("Inconsistent size")
    end
end