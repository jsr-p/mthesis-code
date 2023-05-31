from dstnx.models import ml

if __name__ == "__main__":
    models = ["lgbm", "logreg"]
    targets = [
        "eu_grad",
        "gym_grad",
        "us_grad",
        "real_neet",
        "us_apply",
        "eu_apply",
        "gym_apply",
    ]
    fsets =  [
        "all", 
        "all_excfamily", 
        "all_excneighborhood", 
        "all_exccontrols",
        "fam_only", 
        "controls_only", 
        "neighborhood_only", 
        "gpa_only", 
    ]
    data_suffixes = [
        "radius200_oneyearradius",
        "k50_oneyeark"
    ]
    feature_suffixes = [
        "_oneyearradius200",
        "_oneyeark50"
    ]
    transs = ["", "impute"]
    for model_name in models:
        for target in targets:
            for fs in fsets:
                for data_suffix, feature_suffix in zip(data_suffixes, feature_suffixes):
                    for trans in transs:
                        model = ml.ModelIdentity(
                            model_name,
                            target,
                            fs,
                            group_col="",
                            interaction=False,
                            # trans="impute",
                            trans="",
                            data_suffix=data_suffix,
                            feature_suffix=feature_suffix,
                            conduct_optuna_study=False,
                        )
                        ml.fit(model)