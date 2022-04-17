import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import lightgbm as lgb
import scipy
import scipy.cluster.hierarchy as sch
from functools import partial
from .helper_functions import *
from .utilities import *
from .metrics import *
from .tuning_utils import tune_missing_hyperparameters
from .secondary_utils import n_largest_correlation

def pruner(df, target_column, weight_column=None, pruning_intensity=None, permutation_rounds=1, correlation_method='pearson',
             validation_percent=0.35, validation_column=None, min_gain_percent_for_pruning = 5e-4, params=None,
             objective="rmse", metric=None, additional_metric_kwargs=None, auto_tune_missing_parameters=True,
             random_tuning_runs=10, total_tuning_runs=20, print_tuning_results=False, redundant_threshold=0.99,
             corr_scale_factor=1.0, print_logs=True, seed=42):

    """
    pruner is a pruner of unpredictive features in your model. It incorporates a set of best practices for feature analysis in your modelling life cycle. It effectively uses LightGBM's predictive capabilities along 
    with its variable importance metrics to analyse the best features in your model at various levels of difficulty. pruner also includes some custom heuristics based on general assumptions about Machine Learning modelling.
    pruner works as follows:
     > It first quickly fits a Primary LightGBM model using all variables present in the data,  looks at the variable importance results & removes variables that do not cause a minimum
      `min_gain_percent_for_pruning` of reduction to the training loss
     > If a set of hyperparameters are not specified by the user , it performs hyperparameter tuning automatically & finds the best parameters whose values are not in user inputs(see argument
       `auto_tune_missing_parameters`)
     > Then a main Secondary Model is fit.
     > The permutation importance based on the `metric`(specified by the user) as well as correlation permutations are used to identify features that are highly correlated thus likely useless. These features are removed from further analysis
     > The permutation importance calculated in previous set are run through a set of custom heuristics & returns a recommmendation of features that are found to be not predictive at a particular pruning_intensity(1-5 i.e least strict to very aggressive feature removal) 

    Arguments
    ---------
    df (pandas.DataFrame): DataFrame with the variables to be appraised, the `target_column`, and the `weight_column`
    
    target_column (str): target column name in `df`

    weight_column (str): Observation weight column name in `df` (optional)

    pruning_intensity (int or None): The level of aggression/intensity level when removing features. Either {1, 2, 3, 4, 5, or None}.
    None means return results at all five pruning intensity levels. 1 means the intensity is lower(less variables are removed)
    5 means high intensity & many unpredictive features are removed

    permutation_rounds (int): Number of times permutation importances and correlations are calculated & averaged. For large datasets, set this to 1 
    else it might take a long time. For small datasets, set a high number. Note the higher the value the more accurate results.

    correlation_method (str): Accepts values "pearson", "kendall", or "spearman". Selected method is used to measure the permutated prediction correlation across all features

    validation_percent (float): If `validation_column` is not user specified, the percent of `df`  will be held out as the validation for calculating
    permutation importance and correlation

    validation_column (np.array of booleans or str): A rank-1 array of boolean of values that indicate True for observations in `df` 
    that should be in the validation set (True) and False for ones which should be training set or a string that refers to
    the name of the column in `df` that that represents a boolean Series. This overrides `validation_percent`

    min_gain_percent_for_pruning (float): A floating point number between 0.0 and 1.0 : represents how much percent of total
    gain that a variable must account for in order to be kept for the secondary main model

    params (dict): A dictionary of LightGBM parameters to use for building models. Please see the parameters section of 
    LightGBM documentation for a list of valid parameters and how to specify. If None are provided, defaults are used

    objective (str): The objective function of  models. Supported objectives : "poisson", "gamma", "tweedie",
    "binary", and, "cross_entropy", "rmse", "mae"

    metric (str or callable): If a string, supported metrics : "norm_gini", "rmse", "mae", "tweedie_deviance", "poisson_deviance",
    "gamma_deviance", "cross_entropy", "accuracy", and "fbeta". 
    If a callable function, the function should want to be minimized and not maximized. It should accept rank-1 numpy arrays or 
    pandas Series for predictions, actuals, and weights in that order.

    additional_metric_kwargs (dict): Because supplied metrics may have additional arguments, you can pass these here. tweedie_deviance
    takes a pVal keyword argument, so pass as {"pVal": 1.675}. fbeta takes a `beta` argument so pass as {"beta": 2}

    auto_tune_missing_parameters (bool): If True and if a parameter is missing from `params` argument, parameters will be automatically selected 
    using combination of hueristics from the Primary model results and hyperparameter tuning study. Only these are searched: "max_depth", "reg_lambda", 
    "reg_alpha", "bagging_fraction", "min_gain_to_split", "cat_l2", ,"max_cat_threshold", and "cat_smooth". You can control 
     hyperparamter tuning using `random_tuning_runs` and `total_tuning_runs` arguments

    random_tuning_runs (int): Number of random runs to inform the hyperparameter tuning. These are  exploratory 
    runs and not exploitation. Only will be used if `auto_tune_missing_parameters` is True and important hyperparameters
    are missing from `params`

    total_tuning_runs (int): Total number of trials for the hyperparamater study (including `random_tuning_runs`). Only will 
    be used if `auto_tune_missing_parameters` is True and important hyperparameters are missing from `params`

    print_tuning_results (bool): If `auto_tune_missing_parameters` is True, setting this argument to True will show the results
    of  hyperparameter tuning . 

    redundant_threshold (float): Threshold for determing if two features are redundant due to the linear correlation. If two
    variables have a pearson correlation >= `redundant_threshold` one of the two will be dropped from the analysis

    corr_scale_factor (float): Only for advanced users and for testing. Check https://en.wikipedia.org/wiki/Scaled_correlation 

    print_logs (bool): If True then various messages will be printed as vaperise runs

    seed (int): Random seed for reproducibility

    Retuns
    ------
    dict of four objects
    {
     pruningResults: Importances and vaperise recommendations,
     prediction_correlation: Correlation matrix in the prediction space arrived at by permuting each variable
     cluster_by_pruning_intensity: Cluster assignments at specified pruning_intensity levels,
     model: The LightGBM model used to obtain the other three objects
     }

    """
    msgs = logMessages(print_logs)
  
    np.random.seed(seed)
    dataset_length = len(df)

    # User input error handling
    assert isinstance(df, pd.DataFrame), "`df` is not a pandas DataFrame"
    df_columns = list(df)
    assert isinstance(target_column, str) and target_column in df_columns, "`target_column` not found in `df`"
    assert is_numeric_dtype(df[target_column]), "`target_column` is not a numeric. Convert it to numeric" 
    if weight_column:
        assert isinstance(weight_column, str) and weight_column in df_columns, "`weight_column` cannot be found in `df`"
        assert is_numeric_dtype(df[weight_column]), "`weight_column` is not numeric. Convert it to numeric"
    if pruning_intensity:
        assert isinstance(pruning_intensity, int) and pruning_intensity>0 and pruning_intensity<6, \
        "`pruning_intensity` must be an integer between 1 and 5`"
    if permutation_rounds < 1 or not isinstance(permutation_rounds, int):
        raise Exception("permutation_rounds must be greater than 1 & an integer")
    if correlation_method not in ["pearson", "spearman", "kendall"]:
        raise Exception("correlation_method must be one of 'pearson', 'spearman', 'kendall'")
    if validation_column is not None and (validation_percent <= 0 or validation_percent >= 1):
        raise Exception("validation_percent must be between 0 and 1 ")
    if isinstance(validation_column, str):
        assert validation_column in df_columns, "`validation_column` cannot be found in `df`"
        assert df[validation_column].dtype == "bool", "`validation_column` must be a boolean "

        validation_indicator = df[validation_column].values
        df = df.drop(validation_column, axis=1)
    elif validation_column is not None:
        assert validation_column.dtype == "bool" and len(validation_column) == dataset_length, \
        "`validation_column` should be the same length as `df` and a boolean"
        validation_indicator = validation_column
    else:
        validation_indicator = np.random.choice([True, False], size=dataset_length, p=[validation_percent, (1-validation_percent)])
    if  (min_gain_percent_for_pruning < 0 or min_gain_percent_for_pruning >= 1 or
         not isinstance(min_gain_percent_for_pruning, float)):
         raise Exception("min_gain_percent_for_pruning should be within [0, 1)")
    if params:
        assert isinstance(params, dict), "`params` should be a dictionary"
    if objective not in SUPPORTED_OBJECTIVE_FUNCTIONS:
        raise Exception(f"objective should be one of {SUPPORTED_OBJECTIVE_FUNCTIONS}")
    if isinstance(metric, str):
        if metric not in SUPPORTED_METRICS:
            raise Exception(f"Metric specified is not valid and should be one of {SUPPORTED_METRICS}")
        else:
            metric_to_minimize = METRIC_FOR_MINIMIZATION[metric]
            metric = eval(metric)
    elif metric is None:
        metric_string = OBJECTIVE_METRIC_STRING_LOOKUP.get(objective, objective)
        metric_to_minimize = METRIC_FOR_MINIMIZATION[metric_string]
        metric = OBJECTIVE_METRIC_FUNCTION_LOOKUP[objective]
    elif callable(metric):
        msgs["metric"]()
        metric_to_minimize = True
    else:
        raise Exception(f"Metric specified is not valid & should be one of {SUPPORTED_METRICS} or a callable "
                         "function that takes predictions, actuals and weights in that order")
    if additional_metric_kwargs:
        assert isinstance(additional_metric_kwargs, dict), "`additional_metric_kwargs` should be a dictionary"
        metric = partial(metric, **additional_metric_kwargs)
    assert isinstance(auto_tune_missing_parameters, bool), "`auto_tune_missing_parameters` should be a boolean"
    assert isinstance(random_tuning_runs, int), "`random_tuning_runs` should be an integer"
    assert isinstance(total_tuning_runs, int), "`total_tuning_runs` should be an integer"
    assert isinstance(print_tuning_results, bool), "`print_tuning_results` should be a boolean"
    assert isinstance(print_logs, bool), "`print_logs` should be a boolean"
    assert isinstance(seed, int), "`seed` should be an integer"

    # Reviewing parameters 
    main_parameters = PARAMS.copy() # main parameters are stored here & updated accordingly
    if params is not None:
        user_parameters = {PARAM_ALIASES.get(k, k):v for k,v in params.items()}
    else:
        user_parameters = {}

    # Making sure user specified parameters are given a preference 
    main_parameters.update(user_parameters)
    if "num_leaves" in user_parameters and "max_depth" not in user_parameters:
        main_parameters["max_depth"] = -1
        user_parameters["max_depth"] = -1
    main_parameters["objective"] = objective

    target = df[target_column].values
    weight = df[weight_column].values.copy() if weight_column else np.ones(dataset_length)

    # Converting weight column to float.
    weight = weight.astype(np.float64)
    # Weights are scaled according to the ratio of total data weight to training data weight.
    # This is to ensure if there are any hyperparamters tuned, they will be generalised
    # for the entire dataset
    total_weight_sum = weight.sum()
    training_weight_sum = weight[~validation_indicator].sum()
    weight *= total_weight_sum / training_weight_sum
    
    df = df.drop([target_column, weight_column], errors="ignore", axis=1)

    # drop variables with only one level
    df = df.drop(remove_constant_columns(df), axis=1)

    # Dropping variables that are identically & fully correlated redundant variable that are perfectly correlated. 
    identical_variables = checkFullCorrelation(df,redundant_thresh=redundant_threshold)
    if identical_variables:
        variables_to_drop = []
        for var_set in identical_variables:
            v1, v2 = var_set
            if v2 not in variables_to_drop:
                msgs["redundant"](v1, v2)
            variables_to_drop.append(v2)
        drop_set = set(variables_to_drop)
        df = df.drop(drop_set, axis=1)

    # Object/String variables are converted to categorical    
    categorical_variables = df.dtypes[df.dtypes == "object"].index.to_list()
    categorical_variable_present = False
    if len(categorical_variables) > 0:
        categorical_variable_present = True
        msgs["categorical_variables"](categorical_variables)
        for var in categorical_variables:
            df[var] = df[var].astype("category")
    if np.any(df.dtypes == "category"):
        categorical_variable_present = True

    # Training Primary model to remove weak unpredictive variables. This speeds up the hyperparameter tuning as well in the next stage
    lgb_train, lgb_validation = create_lgb_data(df, target, weight, validation_indicator)
    if main_parameters["max_depth"] != -1:
        NROUNDS_PRIMARYMODEL = Constants.NROUNDS_PRIMARYMODEL//(main_parameters["max_depth"]**2)
    else: 
        NROUNDS_PRIMARYMODEL = Constants.NROUNDS_PRIMARYMODEL//(main_parameters["num_leaves"])
    msgs["primary_model"]()
    primary_model = lgb.train(params=main_parameters, train_set=lgb_train, num_boost_round=NROUNDS_PRIMARYMODEL)
    primary_model_gain = pd.Series(fetch_importance(primary_model, "gain"))
    primary_model_gain /= primary_model_gain.sum()

    # Remove unpredictive variables from Primary model results
    drop_list_primary_model = primary_model_gain[primary_model_gain < min_gain_percent_for_pruning].index.to_list()
    predictive_primary_variables = primary_model_gain[primary_model_gain >=  min_gain_percent_for_pruning].index.to_list()
    drop_count_primary_model = len(drop_list_primary_model)
    predictive_var_count_primary = len(predictive_primary_variables)
    msgs["removeVariablePrimary"](drop_count_primary_model)

    # Prepare data for hyperparameter tuning of user missing parameters
    main_parameters["feature_fraction"] = user_parameters.get("feature_fraction", get_range(predictive_var_count_primary, COLSAMPLE_BYTREE))
    df_secondary_model = df[predictive_primary_variables]
    df_secondary_model._copy = False
    lgb_train, lgb_validation = create_lgb_data(df_secondary_model, target, weight, validation_indicator)

    if auto_tune_missing_parameters:
        # Some parameters are not tuned, hence using direct assignment
        user_parameters["objective"] = objective
        user_parameters["learning_rate"] = main_parameters["learning_rate"]
        user_parameters["feature_fraction"] = main_parameters["feature_fraction"]
        user_parameters["num_leaves"] = main_parameters["num_leaves"]
        user_parameters["bagging_freq"] = main_parameters["bagging_freq"]
        # Check if all parameters are worth tuning
        correct_parameter_check = all(p in user_parameters for p in HYPERPARAMETERTUNING_PARAMS)
        if correct_parameter_check:
            msgs["no_tune"]()
        else:
            msgs["suggest"](total_tuning_runs)
            tuned_parameters = tune_missing_hyperparameters(params=user_parameters, dtrn=lgb_train, dval=lgb_validation,
                                                categorical_variable_present=categorical_variable_present, random_tuning_runs=random_tuning_runs,
                                                total_tuning_runs=total_tuning_runs, 
                                                verbosity=print_tuning_results, seed=seed)
            main_parameters.update(tuned_parameters)

    correct_parameter_check = all(p in user_parameters for p in HYPERPARAMETERTUNING_PARAMS)
    if not correct_parameter_check:
        msgs["hypers"](main_parameters)

    # Training Secondary model to extract correlation of permuted predictions and
    # importances - both permutation and gain. 
    # This is the most time consuming part
    msgs["secondary_Model"]()
    np.random.seed(seed)
    secondary_model = lgb.train(params=main_parameters, train_set=lgb_train, num_boost_round=Constants.NROUNDS,
                           valid_sets=[lgb_train, lgb_validation], valid_names=['trn', 'val'],
                           early_stopping_rounds=Constants.EARLY_STOPPING_ROUNDS,
                           verbose_eval=False)
    
    # Drop varibles that only have one level
    identical_variables_in_validation = remove_constant_columns(lgb_validation.data)
    if identical_variables_in_validation:
        msgs["identical_variables_in_validation_data"](identical_variables_in_validation)
    
    # Drop variables hat were not used in any tree splits
    no_split_variables = fetch_importance(secondary_model, "split")
    drop_list_secondary_model = [v for v,s in no_split_variables.items() if s == 0]
    if drop_list_secondary_model:
        msgs["remove_variables_secondary_model"](drop_list_secondary_model)
    drop_list_secondary_model.extend(identical_variables_in_validation)
    drop_count_secondary = len(drop_list_secondary_model)
    
    msgs["permutation_importance"]()
    _, lgb_validation = create_lgb_data(df_secondary_model, target, weight, validation_indicator)
    correlation, permutation_importance = permutation_importance_calculate(lgb_validation.data, lgb_validation.label, lgb_validation.weight, secondary_model, drop_columns = drop_list_secondary_model,
                                   metric=metric, link_function=MAP_TO_LINK_FUNCTION[objective], runs=permutation_rounds,
                                   importances=True, minimize=metric_to_minimize, correlation=True, correlation_method=correlation_method, print_logs=print_logs)

    cv = correlation.values * corr_scale_factor
    np.fill_diagonal(cv, 1.0)
    correlation.update(pd.DataFrame(cv, columns = correlation.columns, index=correlation.columns))
    gain_importance = fetch_importance(secondary_model, "gain")

    # Creating a result dataframe for collecting importances and correlation clusters
    result_data = pd.DataFrame.from_dict(permutation_importance, orient="index", columns=["permutation_importance"])
    result_data["gain_importance"] = result_data.index.map(gain_importance)
    """
    A noisy data can cause the sum of the permutation importances to be negative which would mess up the whole variable selection logic. 
    Hence we only include those variables that have have positive permutation importances (note that positive permutation importance is considered "good" irrespective of whether 
    the metric is minimised or maximised as that is how it's defined in `permutation_importance_calculate` function). The  effect of this is that 
    the `percent_permutation_importance` will not sum to 1.0 and so the aprior_importance defined in `variables_by_pruning_intensity` is not quite right but negative
    permutation importance variables get filtered out anyway so this is the best option at the moment
    """
    positive_permutation_importance = result_data.permutation_importance > 0
    result_data["percent_permutation_importance"] = result_data.permutation_importance / result_data.permutation_importance[positive_permutation_importance].sum()
    result_data["percent_gain_importance"] = result_data.gain_importance / result_data.gain_importance.sum()

    # Looping through a pre-determined max conephetic distances scaler factors and we store
    # the number of resulting clusters. Smaller scaler values lead to more clusters
	
    dist_scalers = np.arange(0.01, 1.0, step=0.025)
    cluster_temp = np.zeros(shape=(len(dist_scalers), len(list(correlation))))
    corr_distance = sch.distance.pdist(correlation.values)
    maximum_distance = corr_distance.max()
    link = sch.linkage(corr_distance, method="complete")
    for i, d in enumerate(dist_scalers):
        c = sch.fcluster(link, d * maximum_distance, "distance")
        cluster_temp[i,:] = c
        cluster_data = pd.DataFrame(cluster_temp, columns = list(correlation))
    cluster_data['n_unique_clusters'] = cluster_data.nunique(axis=1)
    
    cluster_map_to_intensity = {}
    for intensity in PRUNING_INTENSITY_LOOKUPS.keys():
        # Skipping if pruning_intensity specified
        if pruning_intensity is not None and pruning_intensity != intensity:
            continue
        retain_variable_flag, cluster_result = variables_by_pruning_intensity(result_data, cluster_data, intensity, drop_count_secondary)
        # if retain_variable_flag is a boolean, then post-mapping we get an object because
        # a pandas series can't be boolean? So, retain_variable_flag is 1/0 int64 so we convert
        # to boolean after the mapping.
        result_data[f"removeVariableAtIntensity{intensity}"] = result_data.index.map(retain_variable_flag)
        result_data[f"removeVariableAtIntensity{intensity}"] = result_data[f"removeVariableAtIntensity{intensity}"].astype(np.bool)
        cluster_map_to_intensity[f"clusterAtAggression{intensity}"] = cluster_result
    pruning_data_columns = list(result_data)
    delete_array = np.full(shape=(drop_count_primary_model + drop_count_secondary  , len(pruning_data_columns)), fill_value=np.nan)
    delete_index = drop_list_primary_model + drop_list_secondary_model
    rm_df = pd.DataFrame(delete_array, columns=pruning_data_columns, index=delete_index)
    # set removeVariable to True for dropped columns
    pruning_columns = [col for col in pruning_data_columns if col.startswith("removeVariableAtIntensity")]
    for col in pruning_columns:
        rm_df[col] = True
    final_result = pd.concat([result_data, rm_df], axis=0)
    # sort from highest pruningINtensity to lowest
    sorted_columns = pruning_columns[::-1]
    column_order = [True for _ in sorted_columns]
    # secondary sort on `permutation_importance` column
    sorted_columns.append("permutation_importance")
    column_order.append(False)
    final_result = final_result.sort_values(sorted_columns, ascending=column_order)
    cluster_map_to_intensity_df = pd.DataFrame.from_dict(cluster_map_to_intensity)
    return_dict = {"pruningResults": final_result, "prediction_correlation": correlation, "cluster_by_pruning_intensity": cluster_map_to_intensity_df, "model": secondary_model}
    msgs["done"]()
    return return_dict
    




    

    





    










