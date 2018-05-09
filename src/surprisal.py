from ast import literal_eval
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.spatial.distance import cosine
import spacy
from tqdm import tqdm

NLP = spacy.load("en_core_web_lg")


def clean_punctuation(doc):
    """
    Utility function to clean up a spaCy parsed document
    to have punctuation that matches the Witzels' documents.

    :param doc: spaCy parsed document
    :return: cleaned document
    """
    while True:
        for i in doc:
            if i.text in ("--", "...", ""):
                continue
            elif i.text == "-":
                doc[i.i - 1:i.i + 2].merge()
                break
            elif i.is_left_punct and i.text not in "'’":
                doc[i.i:i.i + 2].merge()
                break
            elif i.is_punct or i.text in("’s", "'s", "n't", "n’t"):
                doc[i.i - 1:i.i + 1].merge()
                break
        # for-else
        else:
            break

    return doc

def load_surprisal(surprisal_file, alpha=1):
    """
    Evaluate the surprisal from discreet probability distributions.
    Probabilities are smoothed additively, according to:
        P(x) = (count(x) + alpha) / (total_word_count + alpha * vocab_size)

    :param surprisal_file: str; path to a .p file containing a dictionary
        to measure surprisal from.
    :param alpha: int or float; smoothing parameter.
    :return: pandas.DataFrame() with tokens and surprisal scores.
    """

    # Load and transform the surprisal metrics
    # into a dictionary.
    probs = pd.read_csv(surprisal_file, compression="bz2")
    probs = probs.set_index("Unnamed: 0")
    probs = probs.astype(np.float32)
    probs = probs.apply(
        lambda x: (x + alpha) / (x.sum() + alpha*len(x)),
        axis=1,
    )
    probs = probs.to_dict(orient="index")
    probs = {
        i.lower().strip():{
            j.lower().strip():probs[i][j]
            for j in probs[i]
            if not pd.isnull(probs[i][j])
        }
        for i in probs
    }
    # Update with the total probabilities of each row.
    for i in probs:
        probs[i]["P_TOT"] = sum(probs[i].values())


    return probs

def evaluate_surprisal(text, surprisal_dict, attr):
    """
    Return a dataframe with columns Token, {attr} Surprisal Score, Metadata

    :param text: text to score. Can be a str or a spaCy-parsed Document object.
    :param surprisal_dict: dictionary returned by load_surprisal.
    :param attr: which spacy attribute to use (pos_, tag_, dep_)
    :return: dataframe
    """
    # if not isinstance(text, spacy.tokens.doc.Doc): text = NLP(text)

    # Calculate the window size from the surprisal dictionary.
    window = max(len(i.split()) for i in surprisal_dict.keys())

    # Get surprisal for each token.
    attrs = [getattr(i, attr).lower().strip() for i in text]
    attrs = [
        (" ".join(attrs[i:i+window]), attrs[i+window])
        for i in range(0, len(attrs) - window)
    ]
    surprisals = [np.nan] * min(window, len(text))
    probabilities = [np.nan] * min(window, len(text))
    for i in attrs:
        tot_prob = -np.log1p(surprisal_dict.get(i[0], {"P_TOT": np.nan})["P_TOT"])
        prob = surprisal_dict.get(i[0], {i[1]:np.nan}).get(i[1], np.nan)
        # prob = surprisal_dict.get(i[0], {i[1]: np.nan})[i[1]]
        surprisals.append(tot_prob - prob)
        probabilities.append(prob)

    # zip up the dataframe and return it
    df = pd.DataFrame({
        "Token":[i.text for i in text],
        "Position":[i.i for i in text],
        attr:[getattr(i, attr) for i in text],
        f"{attr.replace('_', '').capitalize()} Surprisal":surprisals,
        f"{attr.replace('_', '').capitalize()} Probability":probabilities,
    })

    return df

def evaluate_surprisals_continuous(text, window_params, tot_params):
    """
    Get the probability mass functions for each token's
    semantic similarity using a beta distribution.

    :param text: text to analyze; results are returned token-wise
        Can be a string or a spaCy-parsed document.
    :param params: tuple; parameter for scipy.stats.beta distribution
    :return:
    """
    window_dist = beta(*window_params)
    tot_dist = beta(*tot_params)
    # if not isinstance(text, spacy.tokens.doc.Doc): text = NLP(text)

    sims_5 = [np.nan]
    sims_tot = [np.nan]
    for i in range(1, len(text)):
        try:   sims_5.append(cosine(text[max(0, i-5):i].vector, text[i].vector))
        except ZeroDivisionError: sims_5.append(np.nan)

        try:   sims_tot.append(cosine(text[:i].vector, text[i].vector))
        except ZeroDivisionError: sims_tot.append(np.nan)

    # zip up the dataframe and return it
    df = pd.DataFrame({
        "Token":[i.text for i in text],
        "Position":[i.i for i in text],
        "Windowed Similarity":sims_5,
        "Totaled Similarity":sims_tot,
        "Windowed PDF":[-np.log(window_dist.pdf(i)) if not pd.isnull(i) else np.nan for i in sims_5],
        "Totaled PDF":[-np.log(tot_dist.pdf(i)) if not pd.isnull(i) else np.nan for i in sims_tot],
    })

    return df

def main():
    hyphens = re.compile(r"(\S+)-(\S+)")
    ws = re.compile("\s+")
    qs = re.compile(r"“|”")
    apos = re.compile("’")

    print("Loading dependency data.")
    dep = load_surprisal("../out/dep.csv.bz2")
    print("Loading Treebank Tag data.")
    tag = load_surprisal("../out/tag.csv.bz2")
    print("Loading Google POS data.")
    pos = load_surprisal("../out/pos.csv.bz2")
    sim_5 = open("../out/sim-5.txt", "r", encoding="utf8").read()
    sim_5 = literal_eval(sim_5.split("\n")[1])
    sim_tot = open("../out/sim-tot.txt", "r", encoding="utf8").read()
    sim_tot = literal_eval(sim_tot.split("\n")[1])
    stimuli = [
        (i.name, j)
        for i in os.scandir("../stimuli data/split")
        for j in clean_punctuation(
            NLP(apos.sub("'", qs.sub("", ws.sub(" ",
                open(i.path, "r", encoding="utf8").read().strip())))
                )
        ).sents
    ]
    dfs = []
    for i in tqdm(stimuli):
        dep_df = evaluate_surprisal(i[1], dep, "dep_")
        pos_df = evaluate_surprisal(i[1], pos, "pos_")
        tag_df = evaluate_surprisal(i[1], tag, "tag_")
        sim_df = evaluate_surprisals_continuous(i[1], sim_5, sim_tot)
        df = dep_df\
            .merge(pos_df, how="inner", on=["Token", "Position"])\
            .merge(tag_df, how="inner", on=["Token", "Position"])\
            .merge(sim_df, how="inner", on=["Token", "Position"])
        df["Stimulus"] = i[0][:-4]
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv("../out/Surprisals.csv", index=False)
    print("SURPRISAL SHAPE:", df.shape)

if __name__ == "__main__":
    main()