function Y = myModel(X, Parameters)
    disp('Model evaluation')
    Y = Parameters.a .* X; %./ (X + 1.);
end