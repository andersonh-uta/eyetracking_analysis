from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score, f1_score

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
        # replace roc_auc_score with f1_score for multiclass
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