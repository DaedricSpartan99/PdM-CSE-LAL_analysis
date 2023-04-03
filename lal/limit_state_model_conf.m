function out = limit_state_model_conf(PX, Parameters)

    [logL_eval, logL_var] = uq_evalModel(Parameters.LogLikelihood, PX(:,2:end));
    if Parameters.ConfSign == '+'
        sgn = 1;
    elseif Parameters.ConfSign == '-'
        sgn = -1;
    end

    k = norminv(1. - Parameters.ConfAlpha / 2.);

    out = log(PX(:,1)) - Parameters.logC - logL_eval + sgn .* k .* sqrt(logL_var);
end
