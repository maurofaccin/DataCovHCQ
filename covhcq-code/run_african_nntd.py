#!/usr/bin/env python3
"""Decompose tensor with tensorly."""

import json
import pathlib

import numpy as np
import tensorly
from tensorly import decomposition
from tqdm import trange


def strip_keywords(nptensor):
    """Strip keywords used to scrap."""
    with open("../data/ht_cooccurrence_tensor_max20.json", "rt") as fin:
        # load key to index mapping as in tensor
        mapping = {k: int(v) for k, v in json.load(fin)["hashtags"].items()}
    # index to key mapping
    tags = {v: k for k, v in mapping.items()}

    # get keywords
    with open("data/keys-covid.txt", "rt") as fin:
        keys = [k.strip() for k in fin]
    # use indices
    keys = {mapping[k] for k in keys if k in mapping}
    # add few more keys
    keys.update({ind for k, ind in mapping.items() if k[:5] == "covid" and k[-2:] == "19"})

    # invert selection
    keys = [k for k in range(nptensor.shape[0]) if k not in keys]
    # get the new indices
    tags = {i: tags[v] for i, v in enumerate(keys)}

    # remove columns from first 2 dimensions
    print(f"Removing {len(mapping) - len(keys)} hashtags.")
    nptensor = nptensor[keys, :, :]
    nptensor = nptensor[:, keys, :]

    return nptensor, tags


def load_tensor(filepath: str):
    """Load the tensor."""
    nptensor = np.load(filepath)["cooccurrence"]
    return tensorly.tensor(nptensor, dtype="float")


def main(typename):
    """Do the main."""
    tensor = load_tensor(f"../data/ht_cooccurrence_tensor_{typename}.npz")
    print(tensor.shape)

    fileadd = ""
    if False:
        # normalize all adjacency matrices
        tensor /= tensor.sum((0, 1)).reshape((1, 1, -1))
        fileadd += "_norm"

    if False:
        # take the binary version
        tensor = (tensor > 0).astype('float')
        fileadd += "_binary"

    folder = pathlib.Path(f"../data/nntf{fileadd}_{typename}")
    folder.mkdir(exist_ok=True)

    # repeat for number of components
    tcycle = trange(6, 21, 1)
    for comp in tcycle:
        tcycle.set_description(desc=f"{typename} --- p{comp} Computing svd")
        # init with svd
        t_approx, errs = decomposition.non_negative_parafac(
            tensor, comp, return_errors=True, init="svd", n_iter_max=300
        )
        cache = {
            "err": min(errs),
            "a": t_approx.factors[0],
            "b": t_approx.factors[1],
            "c": t_approx.factors[2],
        }

        # test for a random init
        for i in range(10):
            tcycle.set_description(desc=f"{typename} --- p{comp} Computing random init {i}")
            t_approx, errs = decomposition.non_negative_parafac(
                tensor, comp, return_errors=True, init="random", n_iter_max=300
            )
            err = min(errs)
            if cache["err"] > err:
                cache = {
                    "err": err,
                    "a": t_approx.factors[0],
                    "b": t_approx.factors[1],
                    "c": t_approx.factors[2],
                }

        tcycle.set_description(desc=f"{typename} --- Saving")
        np.savez_compressed(
            folder / f"non_negative_parafac_rank_{comp}.npz",
            A=cache["a"],
            B=cache["b"],
            C=cache["c"],
            err=cache["err"],
        )


if __name__ == "__main__":
    for typename in ["geo_early", "geo_late", "nofr", "full", "hcq"]:
        print("computing", typename)
        main(typename)
