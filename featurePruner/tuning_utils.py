from functools import partial
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
from .utilities import Constants

def optimisation_objective(trial, params, dtrn, dval, categorical_variable_present):
    '''
    Function for setting the right parameters for optuna tuning
    '''

    parameters = params.copy()
    parameters["min_gain_to_split"] = parameters.get("min_gain_to_split", trial.suggest_uniform("min_gain_to_split", 0, 25))
    parameters["reg_lambda"] = parameters.get("reg_lambda", trial.suggest_uniform("reg_lambda", 3, 50))
    parameters["reg_alpha"] = parameters.get("reg_alpha", trial.suggest_uniform("reg_alpha", 0.5, 30))
    parameters["bagging_fraction"] = parameters.get("bagging_fraction", trial.suggest_uniform("bagging_fraction", 0.4, 0.9))
    parameters["max_depth"] = parameters.get("max_depth", trial.suggest_int("max_depth", 2, 8))
    parameters["max_cat_threshold"] = parameters.get("max_cat_threshold", 
                                     trial.suggest_int("max_cat_threshold", 5, 50)  if categorical_variable_present else 32)
    parameters["cat_l2"] = parameters.get("cat_l2", trial.suggest_uniform("cat_l2", 1, 40) if categorical_variable_present else 10)
    parameters["cat_smooth"] = parameters.get("cat_smooth", trial.suggest_uniform("cat_smooth", 3, 25) if categorical_variable_present else 10)
    model = lgb.train(params=parameters, train_set=dtrn, num_boost_round=10000, valid_sets=[dtrn, dval],
                    valid_names=["trn", "val"], early_stopping_rounds=Constants.OPTUNA_EARLY_STOPPING_ROUNDS,
                    verbose_eval=False)
    
    validation_metrics = model.best_score["val"]

    val_metric_final = [v for k,v in validation_metrics.items()][0]
    trial.set_user_attr("min_gain_to_split", parameters["min_gain_to_split"])
    trial.set_user_attr("reg_lambda", parameters["reg_lambda"])
    trial.set_user_attr("reg_alpha", parameters["reg_alpha"])
    trial.set_user_attr("bagging_fraction", parameters["bagging_fraction"])
    trial.set_user_attr("max_depth", parameters["max_depth"])
    trial.set_user_attr("max_cat_threshold", parameters["max_cat_threshold"])
    trial.set_user_attr("cat_l2", parameters["cat_l2"])
    trial.set_user_attr("cat_smooth", parameters["cat_smooth"])
    trial.set_user_attr("val_metric", val_metric_final)

    return val_metric_final
    
def tune_missing_hyperparameters(params, dtrn, dval, categorical_variable_present=True, direction="minimize", random_tuning_runs=10,
                         total_tuning_runs=30, verbosity=False, return_all=False, seed=90210):
    '''
    Function to tune missing hyperparameter using optuna
    '''
    optuna.logging.disable_default_handler()
    if verbosity:
        optuna.logging.enable_default_handler()
    sampler = TPESampler(n_startup_trials=random_tuning_runs, seed=seed)
    study = optuna.create_study(sampler=sampler, direction=direction)
    study.optimize(partial(optimisation_objective, params=params, dtrn=dtrn, dval=dval, categorical_variable_present=categorical_variable_present),
                           n_trials=total_tuning_runs)
    trials_df = study.trials_dataframe(multi_index=True)["user_attrs"]
    opt_params = trials_df.iloc[trials_df.val_metric.idxmin()].to_dict()
    _ = opt_params.pop("val_metric", -999)
    int_params = ["max_depth", "max_cat_threshold"]
    for ip in int_params:
        opt_params[ip] = int(opt_params[ip])
    return trials_df if return_all else opt_params





    
