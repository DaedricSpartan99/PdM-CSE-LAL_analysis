function [out, out_var] = limit_state_model(PX, Parameters)

    [logL_eval, logL_var] = uq_evalModel(Parameters.LogLikelihood, PX(:,2:end));
    %logL_eval = uq_evalModel(Parameters.LogLikelihood.Internal.PCE, PX(:,2:end));

    out = log(PX(:,1)) - Parameters.logC - logL_eval;
    out_var = logL_var;
end
