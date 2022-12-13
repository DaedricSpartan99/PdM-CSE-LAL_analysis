function [out, out_var] = limit_state_model(PX, Parameters)

    [logL_eval, logL_var] = uq_evalModel(Parameters.LogLikelihood, PX(:,2:end));

    out = log(PX(:,1)) - log(Parameters.c) - logL_eval;
    out_var = logL_var;
end
