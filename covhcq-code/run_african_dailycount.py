#!/usr/bin/env python3
"""Count tweets on covid and chroroquine."""
import csv
import gzip
import pathlib
import string
import sys
import unicodedata as ucd

import cov_utils
import run_african_tagtensor as att
from tqdm import tqdm

REGIONALINDICATORS = [
    ucd.lookup("REGIONAL INDICATOR SYMBOL LETTER " + c[0]) for c in string.ascii_uppercase
]
REGIONALPATTER = f"[{''.join(REGIONALINDICATORS)}][{''.join(REGIONALINDICATORS)}]"


def log(*args, **kwargs):
    """Log utility."""
    print(*args, **kwargs, file=sys.stderr)


def load_month(month):
    """Load tweets and retweets from month."""
    fname = pathlib.Path(f"../data/african_tweets_{month}.csv.gz")
    with gzip.open(fname, "rt") as fin:
        reader = csv.DictReader(fin)
        yield from reader

    fname = pathlib.Path(f"../data/african_retweets_{month}.csv.gz")
    with gzip.open(fname, "rt") as fin:
        reader = csv.DictReader(fin)
        yield from reader


def alldata():
    """Yield all data from database."""
    keywords = {
        t: cov_utils.get_keywords(t, get_minimal=True, min_len=5)["short"]
        for t in ["hydroxychloroquine", "covid"]
    }  # topics in 'hydroxychloroquine' or 'covid'

    def checktwt(hashtags, tweet, keys):
        for htg in keys:
            if htg in hashtags or htg in tweet:
                return True
        return False

    def check_match(matched):
        if len(matched) == 0:
            return None
        if len(matched) == 1:
            return matched[0]
        return "both"

    all_tweets = {t: set() for t in keywords}
    for month in tqdm(att.__monthlist__("full")):
        for row in load_month(month):
            matched = []

            if "retweeted_id" in row:
                # this is a retweet
                for k in keywords:
                    if row["retweeted_id"] in all_tweets[k]:
                        matched.append(k)
            else:
                # this is a tweet
                for k, kfilter in keywords.items():
                    if checktwt(row.get("hashtags", "").split("|"), row["text"], kfilter):
                        all_tweets[k].add(row["id"])
                        matched.append(k)

            matched = check_match(matched)
            if matched is not None:
                yield matched, row
    log("")


def __compress__(data: dict, delta_days: int = 0):
    if len(data["days"]) <= delta_days:
        return
    remove_day = min(data["days"])
    for k1 in data:
        if k1 == "days":
            continue
        for k2 in data[k1]:
            if isinstance(data[k1][k2].get(remove_day, 0), set):
                data[k1][k2][remove_day] = len(data[k1][k2][remove_day])

    data["days"].remove(remove_day)

    __compress__(data, delta_days=delta_days)


def main():
    """Do main."""
    file_path = pathlib.Path("../data/african_tweets_counts.csv.gz")
    try:
        file_path.unlink()
    except FileNotFoundError:
        pass

    counts = {
        "days": set(),
        "tweets": {"hydroxychloroquine": {}, "covid": {}, "both": {}},
        "retweets": {"hydroxychloroquine": {}, "covid": {}, "both": {}},
    }
    # first detect the users
    log("Build the graph")
    for matched, row in alldata():
        day = row["created_at"][:10]
        # tweet, retweet, user = cov_utils.translate_row(row)

        if "retweeted_id" not in row:
            counts["tweets"][matched].setdefault(day, 0)
            counts["tweets"][matched][day] += 1
        else:
            counts["retweets"][matched].setdefault(day, 0)
            counts["retweets"][matched][day] += 1
        counts["days"].add(day)

        # __compress__(counts, delta_days=60)
    # __compress__(counts, delta_days=2)

    dates = sorted(counts.pop("days"))

    with gzip.open(file_path, "wt") as fout:
        writer = csv.DictWriter(
            fout, fieldnames=["days"] + [f"{k1}_{k2}" for k1, v1 in counts.items() for k2 in v1]
        )
        writer.writeheader()
        writer.writerows(
            [
                dict(
                    days=d,
                    **{
                        f"{trt}_{kw}": counts[trt][kw].get(d, 0)
                        for trt, vals in counts.items()
                        for kw in vals
                    },
                )
                for d in dates
            ]
        )


if __name__ == "__main__":
    main()
