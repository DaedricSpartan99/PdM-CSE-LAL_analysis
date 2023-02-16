function wMed = colwise_weightedMedian(D,W)

    wMed = zeros(1, size(D,2));

    for i = 1:size(D,2)
        wMed(i) = weightedMedian(D(:,i), W);
    end
end