#!/usr/bin/env python3
"""Load the database and write the head and tail marices."""

import csv
import gzip
import pathlib

import cov_afr
import cov_utils
import numpy as np
from scipy import sparse


def load_users():
    """Load user data."""
    with gzip.open("../data/african_tweets_users.csv.gz", "rt") as fin:
        reader = csv.DictReader(fin)
        print(reader.fieldnames)
        data = {row["from_user_id"]: irow for irow, row in enumerate(reader)}

    return data


def load_tweets(users: dict, filter: list = []) -> (dict, sparse.csr_matrix):
    """Load tweets data.

    Parameters
    ----------
    users : dict
        map of user id to int index.
    filter :  list of str, def None
        a list of filters to use: 'hydroxychloroquine', 'covid'…

    Returns
    -------
    tweets : dict
        map if tweet id to int index
    tail : sparse.csr_matrix
        tweets to user incidence matrix
    """
    keywords = {
        t: cov_utils.get_keywords(t, get_minimal=True) for t in filter
    }  # topics in 'vaccine' or 'covid'

    with gzip.open("../data/african_tweets_tweets.csv.gz", "rt") as fin:
        reader = csv.DictReader(fin)
        print(reader.fieldnames)
        if len(keywords) > 0:
            data = {
                row["id"]: users[row["from_user_id"]]
                for row in reader
                if row["from_user_id"] in users
                and cov_utils.filter_row(row, keywords, fmt="any", check_lang=False)
            }
        else:
            data = {
                row["id"]: users[row["from_user_id"]]
                for row in reader
                if row["from_user_id"] in users
            }

    tail = sparse.csr_matrix(
        (
            np.ones(len(data), dtype=np.int64),
            (np.arange(len(data), dtype=np.int64), list(data.values())),
        ),
        shape=(len(data), len(users)),
        dtype=np.int64,
    )

    return {id: iid for iid, id in enumerate(data)}, tail


def load_retweets(users: dict, tweets: dict) -> sparse.csr_matrix:
    """Load reteets and compute the heads."""
    with gzip.open("../data/african_tweets_retweets.csv.gz", "rt") as fin:
        reader = csv.DictReader(fin)
        print(reader.fieldnames)
        data = [
            (tweets[row["retweeted_id"]], users[row["from_user_id"]])
            for row in reader
            if row["retweeted_id"] in tweets and row["from_user_id"] in users
        ]
    head = sparse.csr_matrix(
        (np.ones(len(data), dtype=np.int64), tuple(zip(*data))),
        shape=(len(tweets), len(users)),
        dtype=np.int64,
    )

    return head


def main(kind, month1, month2, name):
    """Do the main."""
    basefile = pathlib.Path(f"../data/african_tweets_{kind}_{name}")

    # load users
    users = list(cov_afr.load_users().keys())

    # load tweets
    if kind == "hydchl":
        tweets, retweets = cov_afr.load_months(
            month1=month1,
            month2=month2,
            filter=["hydroxychloroquine"],
            tweet_keys=["from_user_id"],
            retweet_keys=["from_user_id", "retweeted_id"],
        )
    else:
        tweets, retweets = cov_afr.load_months(
            month1=month1,
            month2=month2,
            tweet_keys=["from_user_id"],
            retweet_keys=["from_user_id", "retweeted_id"],
        )

    print("build the hyperedges")
    tail, head = cov_afr.hyperedges(users, tweets, retweets)

    print("T", tail.sum(), tail.shape, tail.nnz)
    print("H", head.sum(), head.shape, head.nnz)
    sparse.save_npz(basefile.parent / (basefile.name + "_tail.npz"), tail)
    sparse.save_npz(basefile.parent / (basefile.name + "_head.npz"), head)


if __name__ == "__main__":
    kinds = ["full", "hydchl"]
    win = {
        "long": ("2020-01", "2021-10"),
        "early": ("2020-01", "2020-07"),
        "late": ("2020-07", "2021-10"),
    }
    for name, (m1, m2) in win.items():
        print(name, m1, m2)
        for kind in kinds:
            print(kind, "              ", end="\r")
            main(kind, m1, m2, name)
