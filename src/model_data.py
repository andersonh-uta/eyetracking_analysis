import csv
from hashlib import sha1
import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, r2_score

from integration_costs import main as integration_main
from surprisal import main as surprisal_main
import utils

def generate_dataset():
    """
    Generate the main dataset from the surprisal, integration, and eye tracking
    data.  This is the dataset that will be directly used for predictions and
    modeling.

    :return: the dataframe.
    """
    # integration_main()
    # surprisal_main()

    surprisals = pd.read_csv("../out/Surprisals.csv")
    integrations = pd.read_csv("../out/Integration Costs.csv")
    df = surprisals.merge(integrations, how="inner", on=["Token", "Stimulus", "Position"])
    # Covnert stimulus to work for merging with the subject datasets
    df["passage"] = [
        "ARGUIN" if i.startswith("Arguing") else (
        "EDUCAT" if i.startswith("Educating") else (
        "SNAKEH" if i.startswith("Snake Hearts") else (
        "SONGLI" if i.startswith("Songlines") else (
        "BYSTAN" if i.startswith("The Bystander") else (
        "ORIGIN" if i.startswith("The Origin") else (
        "TITLEI"))))))
        for i in df["Stimulus"]
    ]
    df["part"] = [f"Part{i[-1]}" for i in df["Stimulus"]]
    df.to_csv("../out/Surprisals+Integration, merged.csv", index=False)
    df = df[df["passage"] != "SONGLI"]
    print("My DF:", df.shape)

    df = df.rename(columns={"Token":"ia_label", "Position":"IA_ID"})
    df = df.sort_values(by=["passage", "part", "IA_ID"])
    # IDs currently start at 0; they should start at 1
    df["IA_ID"] = df["IA_ID"] + 1
    # replace ellipses with "" character
    df["ia_label"] = [i.replace("...", "…") for i in df["ia_label"]]

    # Read in the subject trials
    df1 = pd.concat(
        pd.read_csv(i.path, encoding="windows-1252", na_values=["."])
        for i in os.scandir("../Subject Data")
        if i.name.endswith(".csv")
    )
    df1 = df1.sort_values(by=["passage", "part", "IA_ID"])
    df1 = df1.reset_index()
    print(df1.shape)
    for i in range(df1.shape[0]):
        if str(df1["ia_label"][i]) == "2000" and int(df1["IA_ID"][i]) == 156:
            df1["ia_label"][i] = "2000."
            print(f"Found \"2000\" --> \"2001.\" change at row {i}.")
    print("Their DF:", df1.shape)

    df2 = df1.merge(df, how="inner", on=["passage", "part", "IA_ID", "ia_label"])
    print("Merged DF:", df2.shape)

    # Load up the participant data file and merge it to the current table.
    participants = pd.concat(
        pd.read_excel("../Subject Data/part_info_template.xlsx", sheet_name=None).values()
    )
    df2 = df2.merge(participants, how="left", left_on="RECORDING_SESSION_LABEL", right_on="ID")

    df2.to_csv("../out/Eye-tracking data with Surprisal+Integration.csv", index=False)

    return df2

def tsne_dataset(df, cols, color_by, suffix=""):
    """
    Perform a TSNE embedding on the eye tracking data.
    Saves results to a file using hashed combinations of
    the data itself and the columns, to avoid re-running
    the time-consuming TSNE if possible.

    :param df: dataframe with data to embed
    :param cols: array-like of str
        columns in the dataframe to use for embedding.
    :param color_by: array-like of str or str
        Column containing values to color on.
    :param suffix: str
        Suffix for the filenames, optional
    :return: generates a plot
    """
    if not os.path.isdir("../out/.tsne"): os.mkdir("../out/.tsne")
    data = df[cols].fillna(0)
    points = data.values
    STR = df.to_string() + " ".join(sorted(cols))

    perplexities = {
        i:sha1((STR+str(i)).encode("utf8")).hexdigest()
        for i in (5, 10, 30, 50)
    }
    for P in perplexities:
        if not os.path.isfile(f"../out/.tsne/{perplexities[P]}.npy"):
            pts = TSNE(perplexity=P, verbose=2, early_exaggeration=40).fit_transform(points)
            np.save(f"../out/.tsne/{perplexities[P]}.npy", pts)


    fignum = 1
    for C in color_by:
        where = ~pd.isnull(df[C])
        cs = dict(map(reversed, enumerate(set(df[C]))))
        cs = [cs[i] for i in df[C][where]]
        plt.figure(fignum)
        sbplt = 1
        for P in perplexities:
            plt.subplot(2, 2, sbplt)
            pts = np.load(f"../out/.tsne/{perplexities[P]}.npy")
            plt.scatter(pts[where,0], pts[where,1], s=1, alpha=.5, c=cs)
            plt.title(f"Perplexity = {P}; colored by {C}")
            sbplt += 1
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"../out/TSNE colored by {C}{suffix}.png", dpi=600)
        # plt.show()
        plt.close("all")

def regression_testing(df, predictors, controls, targets, speaker_split, pos_split, permute=True):
    """
    Run classification models to predict NaN/non-NaN values.

    :param df: pd.DataFrame
        DataFrame containing the data to be analyzed
    :param predictors: array-like of str
        List of column names containing the feature data.
    :param target: array-like of str
        List of column names containing the target variables.
        Multi-output regression will be run on these columns.
    :param split: str
        Used to identify what variable the dataset has been
        split by.
    :return:
    """

    results = {
        "Speaker Split":speaker_split,
        "POS Split":pos_split,
        "Target":[],
        "Control":[],
        "No Control":[],
        "Control Params":[],
        "Control P-Value": [],
        "No Control P-Value": [],
        "No Control Params": [],
        "Chance Score":[],
        "N":[],
    }

    data = df[predictors+controls].fillna(0)
    for T in sorted(targets):
        print(T)
        where = ~pd.isnull(df[T])
        results["Target"].append(T)

        # non-control fit
        r = utils.fit_elasticnet(
            data[predictors][where],
            df[T][where],
            permute=permute
        )
        results["No Control"].append(r[1])
        results["No Control Params"].append(r[0])
        results["No Control P-Value"].append(r[2])
        # control fit
        r = utils.fit_elasticnet(
            data[predictors + controls][where],
            df[T][where],
            permute=permute
        )
        results["Control"].append(r[1])
        results["Control Params"].append(r[0])
        results["Control P-Value"].append(r[2])
        # Compute the by-chance score.
        results["Chance Score"].append(
            r2_score(
                df[T][where].values,
                np.random.permutation(df[T][where].values)
            )
        )
        results["N"].append(len(df[T].fillna(0)))
        print(f"Control score:    {results['Control'][-1]:<.5f}, p={results['Control P-Value'][-1]:<.5f}")
        print(f"No Control score: {results['No Control'][-1]}, p={results['No Control P-Value'][-1]:<.5f}")
        print()

    return pd.DataFrame(results)

def classification_testing(df, predictors, controls, targets, speaker_split, pos_split, permute=True):
    """
    Run classification models to predict NaN/non-NaN values.

    :param df: pd.DataFrame
        DataFrame containing the data to be analyzed
    :param predictors: array-like of str
        List of column names containing the feature data.
    :param target: array-like of str
        List of column names containing the target variables.
        Multi-output regression will be run on these columns.
    :param split: str
        Used to identify what variable the dataset has been
        split by.
    :return:
    """

    results = {
        "Speaker Split":speaker_split,
        "POS Split":pos_split,
        "Target":[],
        "Control":[],
        "No Control":[],
        "Control P-Value":[],
        "No Control P-Value":[],
        "Control Params":[],
        "No Control Params": [],
        "Chance Score":[],
        "N":[],
    }

    data = df[predictors+controls].fillna(0)
    for T in sorted(targets):
        print(T)
        results["Target"].append(T)

        # non-control fit
        try:    r = utils.fit_svc(data[predictors], df[T].fillna(0), permute=permute)
        except: r = ["ERR", "ERR"]
        results["No Control"].append(r[1])
        results["No Control Params"].append(r[0])
        results["No Control P-Value"].append(r[2])
        # control fit
        try:    r = utils.fit_svc(data[predictors + controls], df[T].fillna(0), permute=permute)
        except: r = ["ERR", "ERR"]
        results["Control"].append(r[1])
        results["Control Params"].append(r[0])
        results["Control P-Value"].append(r[2])
        # Compute the by-chance score.
        results["Chance Score"].append(
            roc_auc_score(
                df[T].fillna(0).values,
                np.random.permutation(df[T].fillna(0).values)
            )
        )
        results["N"].append(len(df[T].fillna(0)))
        print(f"Control score:    {results['Control'][-1]:<.5f}, p={results['Control P-Value'][-1]:<.5f}")
        print(f"No Control score: {results['No Control'][-1]}, p={results['No Control P-Value'][-1]:<.5f}")
        print()

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Leys for assumed random variables
    # Keys to split the population on for analysis
    pop_splits = ["pos_"]
    pop_keeps = ["VERB", "NOUN"]
    control_factors = [
        "obj_len",
        "line_from_top",
        "line_from_bottom",
        "total_words_line",
        "line_pos_from_left",
        "line_pos_from_right",
        "launch_site_char",
    ]
    # Keys for assumed predictive factors
    experimental_factors = [
        # Surprisals and probabilities
        'Dep Probability',
        'Dep Surprisal',
        'Pos Probability',
        'Pos Surprisal',
        'Tag Probability',
        'Tag Surprisal',

        # Semantic similarities
        'Totaled PDF',
        'Totaled Similarity',
        'Windowed PDF',
        'Windowed Similarity',

        # Syntactic metrics
        'n_children',
        'n_parents',
        'nearest_child_distance',
        'nearest_parent_distance',
        'n_unconnected'
    ]
    # Keys for target variables to be predicted
    regression_targets = [
        "ff_duration",
        "gaze",
        "gaze_fix_count",
        # "rb_reading",
        "fp_reg_count",
        "total_reg_out_count",
        "total_fix_count",
        "total_time",
    ]
    classification_targets = [
        "IA_SKIP", 
        "fp_reg"
    ]

    # df = generate_dataset()
    df = pd.read_csv(
        "../out/Eye-tracking data with Surprisal+Integration.csv",
        na_values=["."],
        usecols=control_factors 
                + experimental_factors 
                + regression_targets
                + classification_targets
                + pop_splits 
                + ["L1_Dom"],
    )

    # Split into native and non-native speaker
    df["L1_Dom"] = [
        i if i == "english" else "non-english"
        for i in df["L1_Dom"]
    ]


    PERMUTE = False
    # Initial Modeling
    regression_scores = regression_testing(
        df,
        experimental_factors,
        control_factors,
        regression_targets,
        "all",
        "all",
        PERMUTE,
    )
    classification_scores = classification_testing(
        df,
        experimental_factors,
        control_factors,
        classification_targets,
        "all",
        "all",
        PERMUTE,
    )
    for i in df.groupby("pos_"):
        if i[0] not in ("NOUN", "VERB"): continue
        print(f"Groupby POS={i[0]}")
        regression_scores = pd.concat((
            regression_scores,
            regression_testing(
                i[1],
                experimental_factors,
                control_factors,
                regression_targets,
                "all",
                i[0],
                PERMUTE,
            )
        ))
        classification_scores = pd.concat((
            classification_scores,
            classification_testing(
                i[1],
                experimental_factors,
                control_factors,
                classification_targets,
                "all",
                i[0],
                PERMUTE,
            )
        ))
    for i in df.groupby("L1_Dom"):
        print(f"Groupby L1={i[0]}")
        regression_scores = pd.concat((
            regression_scores,
            regression_testing(
                i[1],
                experimental_factors,
                control_factors,
                regression_targets,
                i[0],
                "all",
                PERMUTE,
            )
        ))
        classification_scores = pd.concat((
            classification_scores,
            classification_testing(
                i[1],
                experimental_factors,
                control_factors,
                classification_targets,
                i[0],
                "all",
                PERMUTE,
            )
        ))

    for i in df.groupby(["pos_", "L1_Dom"]):
        if i[0][0] not in ("NOUN", "VERB"): continue
        print(f"Groupby POS={i[0][0]}, L1={i[0][1]}")
        regression_scores = pd.concat((
            regression_scores,
            regression_testing(
                i[1],
                experimental_factors,
                control_factors,
                regression_targets,
                i[0][1],
                i[0][0],
                PERMUTE,
            )
        ))
        classification_scores = pd.concat((
            classification_scores,
            classification_testing(
                i[1],
                experimental_factors,
                control_factors,
                classification_targets,
                i[0][1],
                i[0][0],
                PERMUTE,
            )
        ))

    regression_scores.to_csv("../out/DIFF INTEGRATION Regression Scores.csv")
    classification_scores.to_csv("../out/DIFF INTEGRATION Classification Scores.csv")