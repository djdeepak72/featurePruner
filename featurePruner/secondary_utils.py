import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


__all__ = ["n_largest_correlation", "correlation_heatmap"]

def find_largest_indices(df, n):
    array = np.tril(np.abs(df.values), -1)
    array_flat = array.flatten()
    array_indexes = np.argpartition(array_flat, -n)[-n:]
    array_indexes = array_indexes[np.argsort(-array_flat[array_indexes])]
    return np.unravel_index(array_indexes, array.shape)

def n_largest_correlation(df, n=10):
    """
    Returns the variables and their correlations that have the `n` largest correlation magnitudes in the `df`
    DataFrame

    Arguments
    ---------
    df (pandas.DataFrame): N x N DataFrame of correlations

    n (int): The number of correlations to return. Must be less than the length/width of `df`

    Retuns
    ------
    pandas.DataFrame

    """
    assert df.shape[0] == df.shape[1]

    n = min(len(df)**2, n)
    column_array = df.columns
    # Filling NA's with 0 to ensure no problems in next stages
    df = df.fillna(value=0)
    variable1_index, variable2_index = find_largest_indices(df, n)
    columnA = []
    columnB = []
    corrln = []
    column_name_pair = []
    for var1_index, var2_index in zip(variable1_index, variable2_index):
        column1 = column_array[var1_index]
        column2 = column_array[var2_index]
        if column1 == column2:
            continue
        # Preventing columns from being swapped just to be safe even though de-duping of identical columns happen
        col_name_pair_temp = column1 + column2
        column_name_pair.append(column2 + column1)
        if col_name_pair_temp in column_name_pair:
            continue
        columnA.append(column1)
        columnB.append(column2)
        corrln.append(df.iloc[var1_index, var2_index])
    result_df = pd.DataFrame({"var1": columnA, "var2": columnB, "correlation": corrln})
    return result_df

def correlation_heatmap(df, figsize=(12,8), **kwargs):
    """
    Returns a correlation heatmap from a DataFrame of correlations

    Arguments
    ---------
    df (pandas.DataFrame): N x N DataFrame of correlations

    kwargs (keyword arguments): Passed to seaborn.heatmap()

    Retuns
    ------
    matplotlib.axes object
    """
    assert df.shape[0] == df.shape[1]
    mask = np.triu(df)
    plt.figure(figsize=figsize)
    plt.yticks(rotation = 0)
    
    cbar = kwargs.pop("cbar", True)
    annot = kwargs.pop("annot", True)
    square = kwargs.pop("square", False)
    cmap = kwargs.pop("cmap", "coolwarm")
    vmin = kwargs.pop("vmin", -1)
    vmax = kwargs.pop("vmax", 1)
    
    cbar_kws = kwargs.pop("cbar_kws", {"shrink": 0.3})
    linewidths = kwargs.pop("linewidths", 0.01)
    linecolor = kwargs.pop("linecolor", "black")
    annot_kws = kwargs.pop("annot_kws", {"fontsize": 11})
    fmt = kwargs.pop("fmt", ".1g")

    _ = sns.heatmap(df, mask=mask, cbar=cbar, annot=annot, square=square, cmap=cmap,
                    vmin=vmin, vmax=vmax, cbar_kws=cbar_kws, linewidths=linewidths,
                    linecolor=linecolor, annot_kws=annot_kws, fmt=fmt, **kwargs)
    return _