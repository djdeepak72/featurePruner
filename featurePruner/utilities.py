from .metrics import *
import numpy as np

param_aliases = {
    "learning_rate": ["shrinkage_rate", "eta"],
    "num_leaves": ["num_leaf", "max_leaves", "max_leaf"],
    "bagging_fraction": ["sub_row", "subsample", "bagging"],
    "bagging_freq": ["subsample_freq"], 
    "feature_fraction": ["sub_feature", "colsample_bytree"],
    "reg_alpha": ["lambda_l1"],
    "reg_lambda": ["lambda", "lambda_l2"],
    "min_gain_to_split": ["min_split_gain"],
    "num_leaves": ["num_leaf", "max_leaves", "max_leaf"]
}
PARAM_ALIASES = {item: k for k,v in param_aliases.items() for item in v}

PARAMS = {"learning_rate": 0.04, "max_depth": 4, "num_leaves": 999, "reg_lambda": 5,
          "bagging_fraction": 0.5, "bagging_freq": 1, "reg_alpha": 2, "min_gain_to_split": 0,
          "cat_l2": 10, "feature_fraction": 1.0, "max_cat_threshold": 32, "cat_smooth": 10, 
          "verbosity": -1}

HYPERPARAMETERTUNING_PARAMS = ["reg_lambda", "reg_alpha", "bagging_fraction", "max_depth", "max_cat_threshold",
                 "cat_l2", "cat_smooth", "min_gain_to_split"]

class Constants:
    NROUNDS = 10000
    NROUNDS_PRIMARYMODEL = 4000
    MIN_GAIN_TO_SPLIT_P = 0.025

    EARLY_STOPPING_ROUNDS = 250
    OPTUNA_EARLY_STOPPING_ROUNDS = 100
    REDUNDANT_SEARCH_SAMPLES = 5000
    REDUNDANT_THRESH_BUFFER = 0.04

COLSAMPLE_BYTREE = {
  (1, 20): 0.6,
  (21, 50): 0.4,
  (51, 100): 0.25,
  (101, 150): 0.2,
  (151, 99999999999): 0.1,
  }


def get_range(n_feat, lkup):
    for key, value in lkup.items():
        if key[0] <= n_feat <= key[1]:
            return value

PRUNING_INTENSITY_TO_QUANTILE = {1: 0.85, 2: 0.65, 3: 0.5, 4: 0.3, 5: 0.2}

PRUNING_INTENSITY_LOOKUPS = {1: 0.5, 2: 0.7, 3: 0.9, 4: 1.5, 5: 5}

SUPPORTED_OBJECTIVE_FUNCTIONS = {"rmse", "mae", "poisson", "gamma", "tweedie", "binary", "cross_entropy"}

SUPPORTED_METRICS = {"norm_gini", "rmse", "mae", "tweedie_deviance", "poisson_deviance", "gamma_deviance", "cross_entropy",
           "accuracy", "fbeta"}

OBJECTIVE_METRIC_STRING_LOOKUP = {"poisson": "poisson_deviance", "gamma": "gamma_deviance", "tweedie": "tweedie_deviance",
                       "binary": "cross_entropy"}

OBJECTIVE_METRIC_FUNCTION_LOOKUP = {"rmse": rmse, "mae": mae, "poisson": poisson_deviance, "gamma": gamma_deviance,
                     "tweedie": tweedie_deviance, "binary": cross_entropy, "cross_entropy": cross_entropy}

MAP_TO_LINK_FUNCTION = {
  **dict.fromkeys(["rmse", "mae"], lambda x: x), 
  **dict.fromkeys(["poisson", "gamma", "tweedie"], np.log),
  **dict.fromkeys(["binary", "cross_entropy"], lambda x: np.log(x / (1 - x))),
}

METRIC_FOR_MINIMIZATION = {met: True for met in SUPPORTED_METRICS}
METRIC_FOR_MINIMIZATION["norm_gini"] = False
METRIC_FOR_MINIMIZATION["accuracy"] = False
METRIC_FOR_MINIMIZATION["fbeta"] = False


def logMessages(printLogs=True):
    msgs = {
        "metric": lambda: None,
        "redundant": lambda x,y: None,
        "categorical_variables": lambda x: None,
        "primary_model": lambda: None,
        "removeVariablePrimary": lambda x: None,
        "suggest": lambda x: None,
        "no_tune": lambda : None,
        "hypers": lambda x: None,
        "secondary_Model": lambda: None,
        "identical_variables_in_validation_data": lambda x: None,
        "remove_variables_secondary_model": lambda x: None,
        "permutation_importance": lambda: None,
        "done": lambda: None,
         }
    if printLogs:
        msgs["metric"] = lambda: print(" User specified custom Metric selected. Ensure that the function the inputs is "
                                        "prediction, actual, weight in this same order and also ensure  "
                                        "that its a metric to be minimised & not maximised\n")
        
        msgs["redundant"] = lambda x,y: print(f"Removing variable {y} because it's very highly correlated with {x}\n")
    
        msgs["categorical_variables"] = lambda x: print(f"These string valued columns are being converted to categorical :\n{x}\n")
        
        msgs["primaryModel"] = lambda: print("Training the Primary model to remove unpredictive variables\n")
        
        msgs["removeVariablePrimary"] = lambda x: print(f"Removing {x} variables because they are not predictive "
                                            "according to Primary model results. These will not have any info. ")
											
        msgs["suggest"] = lambda x: print("`autofillMissingParameters` is True so we are going to perform  "
                                            f"hyperparameter tuning for {x} runs \n")
        
        msgs["no_tune"] = lambda: print("No missing parameters were found. Hence skipping hyperparameter tuning of missing parameters \n")
        
        msgs["hypers"] = lambda x: print("Hyperparameters finalised. These are the parameters used in   "
                                         f"Secondary model:\n{x}\n")
        
        msgs["secondary_Model"] = lambda: print("Training of Secondary model has begun after droppin unpredictive variables found in Primary model\n")
        
		
        msgs["identical_variables_in_validation_data"] = lambda x: print(f"Removing variables {x} from further analysis because of only one level in "
                                              "validation data. \n")
											  
        msgs["remove_variables_secondary_model"] = lambda x: print(f"Removing these variables: {x} because they were not used to split trees in he"
                                              "model.\n")
        msgs["permutation_importance"] = lambda: print("Calculating permutation importance & correlation of variables in prediction space "
                                        "space. Please wait since this might take time depending on the size of your data & complexity "
                                        "of the secondary model.\n")
        msgs["done"] = lambda: print("Pruning complete. Result is a dictionary. TO look at results, check the check the 'results' key which has a dataframe as value with all the results. \n")
    
    return msgs