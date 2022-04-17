import pandas as pd
import numpy as np
import lightgbm as lgb
from fastprogress.fastprogress import master_bar, progress_bar
from .utilities import *
from .secondary_utils import *

def permutation_importance_calculate(df, y, w, mod, drop_columns = None, metric = None, link_function = None, runs = 1, importances=True,
               minimize=True, correlation=True, correlation_method='pearson', print_logs=True, seed=90210):
    
    if not importances and not correlation:
        raise Exception("One of 'importances' or 'correlation' argument must be 'True'")
    # Incorporate metric functions
    metric_function = metric if importances else lambda x,y,z: 0.0
    if link_function is None:
        link_function = lambda x: x
    
    np.random.seed(seed)
    
    df = df.copy()
    datasetLength = len(y)
    column_names = list(df)
    if drop_columns:
        column_names = [col for col in column_names if col not in drop_columns]
    total_cols = len(column_names)

    best_prediction = mod.predict(df)
    best_metric_value = metric_function(best_prediction, y, w)
    # predictions from models can be non-linear. Hence applying the link function to the prediction so as to get a linear version of prediction.
    # For example, when modelling binary clasification where you get probability values, applying the link function gives you a log odds which is more linear.
    # This helps to get more accurate correlation values.
    best_prediction = link_function(best_prediction)
    
    correlation_array = np.zeros(shape=(total_cols,total_cols,runs))
    metric_array = np.zeros(shape=(total_cols,runs))
    mb = master_bar(range(runs))
    for i in mb:
        ip1  = i + 1
        permutation_index = np.random.permutation(datasetLength)
        prediction_array = np.zeros(shape=(datasetLength, total_cols))
        metrics = []
        pb = progress_bar(range(total_cols), parent=mb)
        for c in pb:
            mb.child.comment = f"Permuting variables and computing correlation matrix for run {ip1}"
            col = column_names[c]
            original_df = df[col].values.copy()
            new = df.iloc[permutation_index][col].values
            df[col] = new
            # Using linear predictor
            pred = mod.predict(df)
            prediction_array[:, c] = (link_function(pred) - best_prediction)
            metrics.append(metric_function(pred, y, w))
            df[col] = original_df
        mb.main_bar.comment = f"On permuation run {i+1}"

        pred_df = pd.DataFrame(prediction_array, columns=column_names)
        if correlation:
            correlation_array[:,:,i] = pred_df.corr(method=correlation_method).values
        if importances:
            metric_array[:,i] = np.array(metrics) - best_metric_value
    
    if correlation:
        corr_df = pd.DataFrame(correlation_array.mean(axis=2), index=column_names, columns=column_names)
    
    if importances:
        multiplier = 1 if minimize else -1
        perm_importances = metric_array.mean(axis=1) * multiplier
        perm_importances = {col:md for col, md in zip(column_names, perm_importances)}
        perm_importances = {k:v for k,v in sorted(perm_importances.items(), key=lambda x: x[1],
                                              reverse=True)}
    if correlation:
        if importances:
            return corr_df, perm_importances
        else:
            return corr_df
    else:
        return perm_importances

def create_lgb_data(df, y, w, validationIndic):
    
    x_training = df[~validationIndic]
    x_validation = df[ validationIndic]
    
    y_training = y[~validationIndic]
    y_validation = y[ validationIndic]
    
    w_training = w[~validationIndic]
    w_validation = w[ validationIndic]
    
    training = lgb.Dataset(data=x_training, label=y_training, weight=w_training)
    validation = lgb.Dataset(data=x_validation, label=y_validation, weight=w_validation, free_raw_data=False)
    
    return training,validation

def fetch_importance(mod, imptc_type="gain"):
    return {name:val for name,val in 
            zip(mod.feature_name(), mod.feature_importance(imptc_type, mod.best_iteration))}

def importance_to_rank(df, group_col, rank_col):
    return df.groupby(group_col)[rank_col].rank()

def variables_by_pruning_intensity(res_df, cluster_df, intensity_level, n_drops=0):
    df = res_df.copy()

    clustering_threshold = PRUNING_INTENSITY_TO_QUANTILE[intensity_level]
    n_clusters = cluster_df.n_unique_clusters.quantile(clustering_threshold, interpolation="nearest")
    cluster_map = cluster_df[cluster_df.n_unique_clusters == n_clusters].iloc[0]
    _ = cluster_map.pop("n_unique_clusters")
    df["correlation_cluster"] = df.index.map(cluster_map)
    df["permutation_ranks_by_cluster"] = importance_to_rank(df, group_col="correlation_cluster", rank_col="permutation_importance")
    df["gainImportance_ranks_by_cluster"] = importance_to_rank(df, group_col="correlation_cluster", rank_col="gain_importance")
    df["n_cluster"] = df.correlation_cluster.map(df.groupby("correlation_cluster").size())
    """
    Below is the logic for selecting variables by pruning intensity:
    > First, retain all variables that are either the most important permutation or gain importance for each cluster unless
      they are not greater than the apriori feature importance
    > Secondly, retain again all variables that are more important than the multiplier in PRUNING_INTENSITY_LOOKUPS from the intensity_level.
      Importance are determined by either permutation or gain importance
    > Finally filter out all variables that have negative permutation importance
    """
    #  We add  `n_drops` because we want to take into account variables that were not split upon in Primary model.
    apriori_importance = 1 / (len(df) + n_drops)
    importance_multiplier = PRUNING_INTENSITY_LOOKUPS[intensity_level]
    # Ranks are ascending, hence the top ranked variable are equal to the cluster size
    best_cluster_retained_variables = (
                              (df.permutation_ranks_by_cluster == df.n_cluster) | 
                              (df.gainImportance_ranks_by_cluster == df.n_cluster)
                              ) & (
                              (df.percent_permutation_importance >= apriori_importance) |
                              (df.percent_gain_importance >= apriori_importance)
                              )
    good_retained_variables = ((df.percent_permutation_importance > apriori_importance * importance_multiplier) | 
                      (df.percent_gain_importance > apriori_importance * importance_multiplier)
                     )
    positive_importance_filter = df.permutation_importance >= 0

    total_retained_variables = ((best_cluster_retained_variables | good_retained_variables) & positive_importance_filter).values
    df["prune_variable_flag"] = 1
    df.loc[total_retained_variables, "prune_variable_flag"] = 0
    return df["prune_variable_flag"], cluster_map

def checkFullCorrelation(df, redundant_thresh = 0.99, n_samples=Constants.REDUNDANT_SEARCH_SAMPLES):
    datasetLength = len(df)
    n_samples = min(datasetLength, n_samples)
    sample_df = df.sample(n=n_samples)
    # We select column with numerical datatypes since pandas might drop string variables by default.
    #  Kendall's Tau is an appropriate correlation. For the purposes of trees since only order matters 
    # but Kendall's Tau is like 50-60x slower than Pearson's.
    sample_correlations = sample_df.select_dtypes("number").corr()
    # Looking at only the largest 1000 correlations (if that many exist) to speed this up
    top_sample_correlations = n_largest_correlation(sample_correlations, n=1000)
    unwanted = top_sample_correlations.correlation.abs() >= redundant_thresh - Constants.REDUNDANT_THRESH_BUFFER
    unwanted_variable_list = []
    if unwanted.sum() > 0:
        redundant_sample_corr = top_sample_correlations.loc[unwanted, ["var1", "var2"]]
        for row in redundant_sample_corr.itertuples(index=False):
            maybe_redundant_vars = [row.var1, row.var2]
            # Selecting off diagnol
            full_corr = df[maybe_redundant_vars].corr().abs().iloc[1,0]
            if full_corr >= redundant_thresh:
                unwanted_variable_list.append(tuple(maybe_redundant_vars))
    return unwanted_variable_list

def remove_constant_columns(df):
    check_for_single_level = df.nunique(axis=0) == 1
    return df.columns[check_for_single_level].to_list()