#!/usr/bin/env python3
"""Compute inwards fraction of flow."""

import csv

import cov_afr
import numpy as np
from scipy import sparse


def country_projector(no_fr=False):
    """Compute the projector to the country space."""
    users = cov_afr.load_users(extended=True)

    if no_fr:
        countries = {
            u["geo_coding_extened"]
            for u in users.values()
            if len(u["geo_coding_extened"]) == 2 and u["geo_coding_extened"].lower() != "fr"
        }
    else:
        countries = {
            u["geo_coding_extened"] for u in users.values() if len(u["geo_coding_extened"]) == 2
        }

    countries = {u: uind for uind, u in enumerate(sorted(countries))}
    umap = [
        (u["geo_coding_extened"], uind)
        for uind, (uid, u) in enumerate(users.items())
        if u["geo_coding_extened"] in countries
    ]

    proj = sparse.coo_matrix(
        (np.ones(len(umap)), ([u[1] for u in umap], [countries[u[0]] for u in umap])),
        shape=(len(users), len(countries)),
    )

    return proj, users


def map2proj(imap: dict, nusers: int):
    """Compute the map from user space to active user space."""
    proj = sparse.coo_matrix(
        (np.ones(len(imap)), (list(imap.keys()), list(imap.values()))), shape=(len(imap), nusers)
    )
    return proj


def get_comparison(month: str, users: dict, proj: sparse.spmatrix, kind: str = "full") -> dict:
    """Compute the inwards vs outward flow."""
    tail = sparse.load_npz(f"../data/african_tweets_{kind}_{month}_tail.npz")
    head = sparse.load_npz(f"../data/african_tweets_{kind}_{month}_head.npz")

    adj, iumap, _ = cov_afr.adjacency(users, tail, head, tau=-1)

    # use only the users in the largest component (otherwise no steady state)
    iumap = map2proj(iumap, tail.shape[0])
    _proj = iumap @ proj
    adj = (_proj.T @ adj @ _proj).toarray()
    inward = np.diag(adj).sum() / adj.sum()

    # consider only edge count (weighted)
    adj = cov_afr.interaction_matrix(tail, head, tau=-1)
    adj = (proj.T @ adj @ proj).toarray()

    output = {
        "inward": inward,
        "num_users": _proj.shape[0],
        "tot_users": proj.shape[0],
        "num_edges": tail.shape[1],
        "inward_edges": np.diag(adj, k=0).sum(),
        "tot_edges": adj.sum(),
    }

    return {f"{kind}_{k}": v for k, v in output.items()}


def main():
    """Do the main."""
    proj, users = country_projector(no_fr=False)
    print(proj.shape, proj.sum())

    data = []
    for month in cov_afr.cycle_months():
        mdata = get_comparison(month, users, proj)
        mdata.update(get_comparison(month, users, proj, kind="hydchl"))

        data.append(mdata)

    with open("../data/inward_degree.csv", "wt") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    main()
