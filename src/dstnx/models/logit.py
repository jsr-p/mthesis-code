from statsmodels.discrete.discrete_model import LogitResults
import statsmodels.api as sm

from dstnx.models import model_prep


def sm_logit(data, target, group_col):
    endog, exog = prepare_sm(data, target, group_col)
    logit_mod = sm.Logit(endog, exog)
    logit_res: LogitResults = logit_mod.fit(disp=1)
    print("Parameters: ", logit_res.params)
    # margeff = logit_res.get_margeff(  # Slow for many variables
    #     atexog={}
    # )
    # print(margeff.summary())
    print(logit_res.summary())


if __name__ == "__main__":
    target = "eu_grad"
    data = model_prep.load_data()
    endog, exog = model_prep.prepare_sm(data, target)
    sm_logit(data, target=target, group_col="KOM")
    # sm_logit(data, target="gym_grad", group_col="KOM")
    # sm_logit(data, target="eu_grad", group_col="INSTNR")
    # sm_logit(data, target="gym_grad", group_col="INSTNR")
