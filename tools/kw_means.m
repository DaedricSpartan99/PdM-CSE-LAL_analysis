function [labels, centroids] = kw_means(X, W, k, varargin)

    %% Default values
    n_iter = 20;

    %% Manage varargin
    if nargin > 3
        n_iter = varargin{1};
    end

    %% Assign initial clusters
    p = randperm(length(X));
    c_ind = p(1:k);
    centroids = X(c_ind, :);

    old_labels = zeros(length(X),1);
    labels = ones(length(X),1);
    iter = 1;

    while any(labels ~= old_labels) && iter <= n_iter  

    %% Assign values to centroids
        kD = pdist2(X, centroids);%, 'cityblock');
        old_labels = labels;
        [~, labels] = min(kD,[],2);
    
    %% Compute new centroids
        for i = 1:k
            x_i = X(labels == i,:);
            w_i = W(labels == i);
            %if isempty(w_i) 
            %    centroids(i,:) = NaN;
            %else
            %    centroids(i,:) = colwise_weightedMedian(x_i, w_i); %sum(w_i .* x_i) / sum(w_i);
            %end

            centroids(i,:) = sum(w_i .* x_i) / sum(w_i);
        end
    end
end