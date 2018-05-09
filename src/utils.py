"""
MIscellaneous utility functions.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, permutation_test_score
from sklearn.ensemble import \
    RandomForestRegressor, \
    RandomForestClassifier, \
    AdaBoostRegressor, \
    AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.linear_model import HuberRegressor, ElasticNetCV
from scipy.stats import norm, beta

def flatten(it):
    for i in it:
        if hasattr(i, "__iter__") and not isinstance(i, str):
            yield from flatten(i)
        else:
            yield i

"""
Functions to fit various sklearn models.
"""
def fit_rfr(data, targets, permute=True):
    """
    Random forest regressor
    """
    cv = RandomizedSearchCV(
        RandomForestRegressor(n_estimators=50),
        param_distributions={
            "max_depth":np.append(np.arange(5, 50), None),
            "min_samples_split":np.arange(2, 15),
            "min_samples_leaf":np.arange(1, 10),
            "max_features":np.arange(1, data.shape[1]),
        },
        n_jobs=3,
        error_score=0,
        n_iter=100,
        cv=3,
        verbose=1,
    )
    cv.fit(data.values, targets)
    if permute == True:
        p = permutation_test_score(
            cv,
            data,
            targets,
            # cv=10,
            n_jobs=3,
            n_permutations=1000,
        )
        return cv.best_params_, cv.best_score_, p[-1]
    else:
        return cv.best_params_, cv.best_score_, -1


def fit_rfc(data, targets, permute=True):
    """
    Random forest classifier
    """
    cv = RandomizedSearchCV(
        RandomForestClassifier(n_estimators=50),
        param_distributions={
            "max_depth": np.append(np.arange(5, 50), None),
            "min_samples_split": np.arange(2, 15),
            "min_samples_leaf": np.arange(1, 10),
            "max_features": np.arange(1, data.shape[1]),
        },
        n_jobs=3,
        error_score=0,
        n_iter=100,
        verbose=1,
        cv=3,
        scoring=make_scorer(roc_auc_score)
    )
    cv.fit(data.values, targets)
    if permute == True:
        p = permutation_test_score(
            cv,
            data,
            targets,
            # cv=10,
            n_jobs=3,
            n_permutations=1000,
        )
        return cv.best_params_, cv.best_score_, p[-1]
    else:
        return cv.best_params_, cv.best_score_, -1
def fit_huber(data, targets, permute=True):
    """
    Huber regression
    """
    cv = GridSearchCV(
        HuberRegressor(),
        param_grid={
            "epsilon":np.linspace(1, 3, 20),
            "alpha":np.logspace(-10, 0, 10),
        },
        n_jobs=3,
        error_score=0,
        verbose=0,
        cv=3,
    )
    cv.fit(data.values, targets)
    if permute == True:
        p = permutation_test_score(
            cv,
            data,
            targets,
            # cv=10,
            n_jobs=3,
            n_permutations=1000,
        )
        return cv.best_params_, cv.best_score_, p[-1]
    else:
        return cv.best_params_, cv.best_score_, -1

def fit_elasticnet(data, targets, permute=True):
    """
    Elasticnet regression
    """
    cv = ElasticNetCV()
    cv.fit(StandardScaler().fit_transform(data.values), targets)
    params = {"alpha":cv.alpha_, "l1_ratio":cv.l1_ratio_}
    score = cv.score(StandardScaler().fit_transform(data.values), targets)
    if permute == True:
        p = permutation_test_score(
            cv,
            data,
            targets,
            # cv=10,
            n_jobs=3,
            n_permutations=1000,
        )
        return params, score, p[-1]
    else:
        return params, score, -1

def fit_svc(data, targets, permute=True):
    """
    Huber regression
    """
    cv = GridSearchCV(
        LinearSVC(dual=False),
        param_grid={
            "C":np.logspace(-10,5,16),
        },
        n_jobs=3,
        error_score=0,
        scoring=make_scorer(roc_auc_score, average="weighted"),
        verbose=1,
        cv=3,
    )
    cv.fit(StandardScaler().fit_transform(data.values), targets)
    if permute == True:
        p = permutation_test_score(
            cv,
            data,
            targets,
            # cv=10,
            n_jobs=3,
            n_permutations=1000,
        )
        return cv.best_params_, cv.best_score_, p[-1]
    else:
        return cv.best_params_, cv.best_score_, -1

def fit_svr(data, targets, permute=True):
    """
    Huber regression
    """
    cv = GridSearchCV(
        LinearSVR(dual=False, loss="squared_epsilon_insensitive"),
        param_grid={
            "C":np.logspace(-10,5,16),
            "epsilon":np.logspace(-10,5,16),
        },
        n_jobs=3,
        error_score=0,
        verbose=1,
        cv=3,
    )
    cv.fit(StandardScaler().fit_transform(data.values), targets)
    if permute == True:
        p = permutation_test_score(
            cv,
            data,
            targets,
            # cv=10,
            n_jobs=3,
            n_permutations=1000,
        )
        return cv.best_params_, cv.best_score_, p[-1]
    else:
        return cv.best_params_, cv.best_score_, -1