function Y = myModel(X, Parameters)
    disp('Model evaluation')
    Y = Parameters.a .* X;
end