function [labels, centroids] = w_means(X, W, k_range, varargin)

    %% Cross validate correct k
    F_score = zeros(length(k_range),1);
    overall_mean = sum(W .* X) / sum(W);

    for k = k_range

        % perform weighted k-means
        [labels, centroids] = kw_means(X, W, k, varargin{:});

        % take label counts
        [~,counts] = count_unique(labels);

        % compute square euclidean distances from overall mean
        F_sqdist = sum((centroids - overall_mean).^2, 2);

        % compute score
        F_score(k - k_range(1) + 1) = sum(counts .* F_sqdist) / (k-1);
    end

    %% Find F_score curve elbow and apply actual weighted mean clustering
    [~, opt_k_ind] = knee_pt(k_range, F_score);
    opt_k = k_range(opt_k_ind);

    [labels, centroids] = kw_means(X, W, opt_k, varargin{:});
end