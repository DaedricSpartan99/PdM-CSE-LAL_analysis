function out = limit_state_model(XP, Parameters)

    L = uq_evalModel(Parameters.Likelihood, XP(:,1:end-1));

    out = log(XP(:,end)) - log(Parameters.c) - log(L + Parameters.epsilon);
end
