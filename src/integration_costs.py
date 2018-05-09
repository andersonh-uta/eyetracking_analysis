from collections import namedtuple
import os
import re

import numpy as np
import pandas as pd
import spacy

def flatten(it):
    """
    Flatten an arbtirarily deeply nested iterable.

    :param it: iterable to flatten
    :return: generator that flattens the iterable.
    """
    for i in it:
        if hasattr(i, "__iter__") and not isinstance(i, str):
            yield from flatten(i)
        else:
            yield i

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
                doc[i.i-1:i.i+2].merge()
                break
            elif i.is_left_punct and i.text not in "'’":
                doc[i.i:i.i+2].merge()
                break
            elif i.is_punct or i.text in("’s", "'s", "n't", "n’t"):
                doc[i.i-1:i.i+1].merge()
                break
        # for-else
        else:
            break

    return doc

def trace_children(token):
    """
    Calculate the length of the chain formed by successive
    children dependencies.  Works recursively.

    :param token:
    :param positions:
    :return:
    """
    for i in token.children:
        if i.is_punct or i.is_space:
            yield from trace_children(i)
        else:
            yield 1, trace_children(i)

        # elif i.i < token.i:
        #     yield 1, trace_children(i)
        # elif i.i > token.i:
        #     yield 0

def trace_parents(token):
    """
    Calculate the length of the chain formed by successive
    head dependencies.  Works recursively.

    :param token:
    :param positions:
    :return:
    """
    CUR = token.head
    old = -1
    if CUR.i >= token.i:
        return 0
    n_tokens = 0
    # while CUR.i < token.i and CUR.i != old:
    while CUR.i != old:
        # if not (CUR.is_punct or CUR.is_space):
        #     n_tokens += 1
        n_tokens += 1
        old = CUR.i
        CUR = CUR.head


    return n_tokens

def integration_costs(doc, nlp):
    """
    Returns each token in order, along with:
        - The number of seen items which are connected below the current item
        - The number of seen items which are connected above the current item
        - The number of seen items which are not yet connected to the current item
        - The distance between the item and its next immediate child
        - The distance between the item and its next immediate head.

    The text is chunked into sentences with spaCy's automatic parser.
    Each sentence is analyzed independently, then the results
    are concatenated back into the original order.

    :param text: spaCy document.
    :return: a list of tuples.
    """

    Token = namedtuple(
        "Token",
        [
            "Token",
            "n_children",
            "n_parents",
            "nearest_child_distance",
            "nearest_parent_distance",
            "n_unconnected",
            "Position",
        ],
    )
    scored_document = []
    for SENT in doc.sents:
        # dict of {spaCy token: position in sentence},
        # only counting non-punctuation, non-whitespace tokens
        # SENT = [
        #     i
        #     for i in SENT
        #     if not (i.is_punct or i.is_space)
        # ]
        # print(SENT)
        positions = dict(map(reversed, enumerate(SENT)))

        # Number of connected children currently observed
        n_children = [
            sum(flatten(trace_children(i)))
            for i in SENT
        ]

        # number of connected parents currently observed
        n_parents = [
            trace_parents(i)
            for i in SENT
        ]

        # Number of observed tokens with no connection to the
        # current one
        n_unconnected = [
            positions[SENT[i]] - n_children[i] - n_parents[i]
            for i in range(len(SENT))
        ]

        # Distance to the nearest child; NaN if no child.
        nearest_child = [
            positions[i] - max(positions.get(j, np.nan)
                               for j in i.children
                               # if not (j.is_punct or j.is_space)
                               # and j.i < i.i
                               if j.i < i.i
                               )
            if any(
                # not (j.is_punct or j.is_space)
                # and j.i < i.i
                j.i < i.i
                for j in i.children
            )
            else np.nan
            for i in SENT
        ]

        # Distance to the token's syntactic head; NaN
        # if not yet observed.
        head_distance = [
            # positions[i] - positions[i.head]
            i.i - i.head.i
            for i in SENT
        ]
        # head_distance = [
        #     i if i >= 0 else np.nan
        #     for i in head_distance
        # ]

        for i in range(len(SENT)):
            scored_document.append(Token(
                Token=SENT[i].text,
                n_children=n_children[i],
                n_parents=n_parents[i],
                nearest_child_distance=nearest_child[i],
                nearest_parent_distance=head_distance[i],
                n_unconnected=n_unconnected[i],
                # Position=positions[SENT[i]],
                Position=SENT[i].i,
                # n_chunks=n_chunks[i],
            ))
    return pd.DataFrame(scored_document)


def main():
    nlp = spacy.load("en_core_web_lg")
    print("spaCy model loaded.")
    # hyphens = re.compile(r"(\S+)-(\S+)")
    ws = re.compile("\s+")
    qs = re.compile(r"“|”")
    apos = re.compile("’")

    docs = {
        i.name:clean_punctuation(
            nlp(apos.sub("'", qs.sub("", ws.sub(" ",
                open(i.path, "r", encoding="utf8").read().strip())))
                )
        )
        for i in os.scandir("../stimuli data/split")
    }

    # Merge punctuation to the proper adjacent token
    # to match the tokenization in the eye-tracking datasets.
    # for DOC in docs:
    #     for TOK in DOC:
    #         if TOK.is_left_punct:
    #             DOC[TOK.i:TOK.i+2].merge()

    integration = {
        i:integration_costs(docs[i], nlp)
        for i in docs
    }
    for i in integration:
        integration[i]["Stimulus"] = i[:-4]
    integration = pd.concat(list(integration.values()))
    # integration = integration.reset_index()
    integration.to_csv("../out/Integration Costs.csv", index=False)
    print("INTEGRATION SHAPE:", integration.shape)

if __name__ == "__main__":
    main()