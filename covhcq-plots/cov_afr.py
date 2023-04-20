#!/usr/bin/env python3
"""Utility functions for the Country aware analysis."""

import csv
import gzip
from itertools import count

import cov_utils
import numpy as np
from scipy import sparse


def load_months(
    month1: str = "2020-01",
    month2: str = "2021-10",
    filter: list = [],
    tweet_keys: dict = None,
    retweet_keys: dict = None,
):
    """Load Tweets and retweets for the given months."""
    tweets = {}
    retweets = {}

    def sub_keys(row, keys):
        if keys is None:
            return row
        return {k: row[k] for k in keys}

    keywords = {
        t: cov_utils.get_keywords(t, get_minimal=True) for t in filter
    }  # topics in 'vaccine' or 'covid'

    for month in cycle_months(month1=month1, month2=month2):
        # load tweets
        print("Loading tweets")
        with gzip.open(f"../data/african_tweets_{month}.csv.gz", "rt") as fin:
            reader = csv.DictReader(fin)
            if len(keywords) == 0:
                tweets.update({t["id"]: sub_keys(t, tweet_keys) for t in reader})
            else:
                tweets.update(
                    {
                        t["id"]: sub_keys(t, tweet_keys)
                        for t in reader
                        if cov_utils.filter_row(t, keywords, fmt="any", check_lang=False)
                    }
                )

        # load retweets
        print("Loading retweets")
        with gzip.open(f"../data/african_retweets_{month}.csv.gz", "rt") as fin:
            reader = csv.DictReader(fin)
            # we load tweets in a sorted fashion so all original tweets should preceed the retweets.
            # unless weird things happens like time zone shifts.
            retweets.update(
                {t["id"]: sub_keys(t, retweet_keys) for t in reader if t["retweeted_id"] in tweets}
            )

    return tweets, retweets


def load_users(extended=False):
    """Load user data."""
    with gzip.open("../data/african_users.csv.gz", "rt") as fin:
        reader = csv.DictReader(fin)
        data = {row["from_user_id"]: row for row in reader}

    if extended:
        with gzip.open("../data/african_users_extended.csv.gz", "rt") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                data[row["from_user_id"]]["geo_coding_extened"] = row["geo_coding_extened"]

    return data


def cycle_months(month1: str = "2020-01", month2: str = "2021-10"):
    """Cycle on the months.

    Last month excluded.
    """
    year, month = map(int, month1.split("-"))
    for step in count(0):
        __month__ = f"{year}-{month:02d}"
        if __month__ == month2:
            break

        yield __month__

        month += 1
        if month == 13:
            year += 1
            month = 1

        if step > 100:
            raise AttributeError(f"Check attributes: {month1} and {month2}")


def hyperedges(users: list, tweets: dict, retweets: dict):
    """Compute the hyperedges matrices."""
    # save a ordered map to indexes
    nodemap = {uid: iuser for iuser, uid in enumerate(users)}

    retweeted = {}
    for rtw in retweets.values():
        retweeted.setdefault(rtw["retweeted_id"], []).append(rtw["from_user_id"])

    tails = sparse.coo_matrix(
        (
            np.ones(len(retweeted)),
            (
                [nodemap[tweets[tid]["from_user_id"]] for tid in retweeted],
                np.arange(len(retweeted)),
            ),
        ),
        shape=(len(nodemap), len(retweeted)),
    )

    heads = sparse.coo_matrix(
        (
            np.ones(sum([len(r) for r in retweeted.values()])),
            (
                [nodemap[uid] for rtws in retweeted.values() for uid in rtws],
                [tind for tind, rtws in enumerate(retweeted.values()) for _ in rtws],
            ),
        ),
        shape=(len(nodemap), len(retweeted)),
    )
    return tails.tocsc(), heads.tocsc()


def interaction_matrix(tails, heads, tau=-1):
    """Compute the weights of the effective adjacency matrix.

    Only the corresponding transition matrix should be considered.

    Parameters
    ----------
    tails : sparse matrix

    heads : sparse matrix

    tau : {-1, 0, 1}
        parameter
        tau = 0 -> project each hyper_edge to a clique
        tau = -1 -> each hyper edge is selected with the same prob (independently from cascade size)
        tau = 1 -> hyper edges with larger cascades are more probable.
        (Default value = -1)

    Returns
    -------
    adjacency matrix : sparse
    """
    # B_{\alpha, \alpha}
    # get the exit size of each hyper edge (number of vertices involved)
    hyper_weight = heads.sum(0).A1

    # here we may have zeros entries for tweets that have never been retweeted
    hyper_weight[hyper_weight > 0] = hyper_weight[hyper_weight > 0] ** tau
    # put that on a diagonal matrix
    hyper_weight = sparse.diags(hyper_weight, offsets=0)

    # compute the tails -> heads weighted entries (propto probability if symmetric)
    return tails @ hyper_weight @ heads.T


def find_components(matrix, kind="strong"):
    """Return the components of the graph.

    Parameters
    ----------
    matrix : sparse.spmatrix
        the adjacency square matrix
    kind : str, default=`strong`
        either `strong` or `weak` (Default value = 'strong')

    Returns
    -------
    components : list
        sorted list of components (list of node indexes)
    """
    # check strongly connected component
    ncomp, labels = sparse.csgraph.connected_components(
        csgraph=matrix, directed=True, connection=kind
    )

    components = [[] for _ in range(ncomp)]
    for node, label in enumerate(labels):
        components[label].append(node)

    return sorted(components, key=len, reverse=True)


def extract_components(matrix: sparse.spmatrix, indexes: list):
    r"""Extract the sub matrix.

    Parameters
    ----------
    matrix : sparse.spmatrix
        the matrix (square)
    indexes : list
        list of indeces to retain

    Returns
    -------
    matrix : sparse.csc_matrix
        matrix with rows and columns removed.
    """
    return matrix.tocsr()[indexes, :].tocsc()[:, indexes]


def compute_transition_matrix(matrix, return_steadystate=False, niter=10000):
    r"""Return the transition matrix.

    Parameters
    ----------
    matrix : sparse.spmatrix
        the adjacency matrix (square shape)
    return_steadystate : bool (default=False)
        return steady state. (Default value = False)
    niter : int (default=10000)
        number of iteration to converge to the steadystate. (Default value = 10000)

    Returns
    -------
    trans : np.spmatrix
        The transition matrix.
    v0 : np.matrix
        the steadystate
    """
    # marginal
    tot = matrix.sum(0).A1
    # fix zero division
    tot_zero = tot == 0
    tot[tot_zero] = 1
    # transition matrix
    trans = matrix @ sparse.diags(1 / tot)

    # fix transition matrix with zero-sum rows
    trans += sparse.spdiags(tot_zero.astype(int), 0, *trans.shape)

    if return_steadystate:
        v0 = matrix.sum(0)
        v0 = v0.reshape(1, matrix.shape[0]) / v0.sum()
        for i in range(niter):
            # evolve v0
            v1 = v0.copy()
            v0 = v0 @ trans.T
            if np.sum(np.abs(v1 - v0)) < 1e-7:
                break
        print(f"TRANS: performed {i} itertions.")

        return trans, v0

    return trans


def adjacency(
    users: dict,
    tweets: dict,
    retweets: dict,
    tau: float = -1,
    fix_sources="basin",
    return_factors=False,
    symmetrize_weight=0.1,
):
    r"""Return the adjacency matrix of the data.

    It picks the strongly connected component and return

    .. math::
        \Pi T

    where :math:`\Pi` is the diagonal matrix of the steadystate and
    :math:`T` is the transition matrix.
    In this way all edge weights are the probability of being traversed.

    Parameters
    ----------
    data : DataBase or tuple
        database of tweets or tuple(user2id_map, tails heads)
    tau : int
        parameter.
        (Default value = -1)
    fix_sources : bool, default=True
        if a fix to source and sink nodes need to be applied.
        This will performed adding a fake `basin` node as a bridge between sink and source nodes.
        It will be removed before return. (Default value = True)
    return_factors :
         (Default value = False)

    Returns
    -------
    adjacency : sparse
        the adjacency matrix
    umap : dict
        map of index to user IDs
    other_components : list of lists
        list of components other that the largest.
    """
    # compute hyper_edges (tails and heads)
    if isinstance(tweets, dict) and isinstance(retweets, dict):
        tails, heads = hyperedges(users, tweets, retweets)
    else:
        tails, heads = tweets, retweets
    iu_map = {i: u for i, u in enumerate(users)}

    # put everything in a matrix (interaction matrix)
    weighted_adj = interaction_matrix(tails, heads, tau=tau)
    weighted_adj += symmetrize_weight * weighted_adj.T

    # extract the largest connected component
    comps = find_components(weighted_adj, kind="strong")
    assert sum([len(c) for c in comps]) == weighted_adj.shape[0]
    weighted_adj = extract_components(weighted_adj, comps[0])

    # compute the transition matrix and the steady state.
    transition, steadystate = compute_transition_matrix(
        weighted_adj,
        return_steadystate=True,
        niter=10000,
    )
    del weighted_adj
    steadystate = steadystate.A1
    assert len(comps[0]) == transition.shape[0]
    transition.eliminate_zeros()

    print(
        f"STRONG COMPONENT: we discarted {tails.shape[0] - transition.shape[0]}"
        f" nodes {100 * (tails.shape[0] - transition.shape[0]) / tails.shape[0]:4.2f}%."
    )
    print(f"STRONG COMPONENT: adjacency matrix of shape {transition.shape}")
    if return_factors:
        return (
            transition,
            steadystate,
            {i: iu_map[cind] for i, cind in enumerate(comps[0])},  # the node IDs
            [[iu_map[i] for i in comp] for comp in comps[1:]],  # nodes discarted
        )

    # new adjacency matrix as probability A_{ij} \propto p(i, j)
    new_adj = transition @ sparse.diags(steadystate)
    return (
        new_adj,  # the adjacency matrix
        {i: cind for i, cind in enumerate(comps[0])},  # the node IDs
        [[iu_map[i] for i in comp] for comp in comps[1:]],  # nodes discarted
    )
