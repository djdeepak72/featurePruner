![](Pruner.png)

`featurePruner` is a Python feature selection package.. It provides recommendations on whether to remove (prune) or retain a variable for your various predictive modelling tasks. It uses permutation importance at its core, some custom clustering heuristics, LightGBM & couple of other best practives to inform you of the most predictive variables in your data.

### Installation

To install from cmd terminal run:
`pip install git+https://github.com/willofdeep/featurePruner.git`

OR

Clone the repo, cd into the repo directory & run one of the below commands:

`pip install .`

OR

`python setup.py install`


### What does this package do?
`featurePruner` has a main function `pruner`, which provides recommendations to the best variables in your model to predict your response variable. `pruner` uses gradient boosted decision trees from the LightGBM package to derive these variable recommendations. Since LightGBM is a very fast model building library, it can work with missing data, it handles both ordinal and categorical variables, and GBDTs are flexible enough to work for linear as well as highly non-linear modeling problems. `pruner` encompasses a set of best practices and works as follows:

* Fit an Primary model to quickly analyse the variables and discard variables that don't contribute at least `min_gain_percent_for_pruning` percent reduction to the training's `objective`. We recommend that this model be fit with the hyperparameter `feature_fraction = 1.0` (the default) so that each variable's contribution is based on all variables being present

* Prepare setup to build a secondary model by optionally tuning the model's hyperparameters. This occurs if `auto_tune_missing_parameters = True` and not all hyperparameters are specified in the `params` argument

* Fit a secondary model with the remaining variables.

* Conduct the variable importance analysis by randomly permuting each variable, calling the secondary model to get the predictions, and then (a) computing the difference in the `metric` between the predictions with and without the variable permutation and (b) computing the difference in the predictions themselves with and without the variable permutation. `pruner` then computes the correlations across the variables based on the vector of prediction differences in (b). Finally, this correlation matrix is fed into a hierarchal clustering procedure to determine which variables are highly correlated in the prediction space.

* Based on the `pruning_intensity` levels specified by the user, the optimal number of clusters are determined (a higher `pruning_intensity` means fewer clusters and vice versa)

* Using the variables' `metric` permutation differences, the variables' gains, and the clustering assignment a set of heuristic rules is applied  to determine whether to remove or retain a variable. In addition to determining the number of clusters, the `pruning_intensity` also uniquely defines the set of heuristic rules. Lower `pruning_intensity`(s) remove less variables, and high `aggression_level`(s) remove more variables

* `pruner` returns a pandas DataFrame where the index (rows) are the variables and the columns are the permutation and gain importances as well as whether the variable should be removed at a particular `pruning_intensity`

Since leaving too many decisions to the user leads to paralysis by analysis, and so `featurePruner` tries to guide the variable importance analyses by making some hard decisions. At the same time, featurePruner does allow flexibility by allow for an appraisel to run at any of five pruning_intensity levels (or even all five at once) and returns back the results that help explain its recommendations so the user can understand how it came to those decisions

### Want to know more?
For more info about this package, please see the [FAQ](FAQ.md) and the [beginner's tutorial](docs/Example.ipynb)


