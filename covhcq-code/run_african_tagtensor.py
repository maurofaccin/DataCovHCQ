#!/usr/bin/env python3
"""Extract the tags co-occurrence (temporal) tensor."""

import csv
import gzip
import json
import pathlib
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

import cov_utils
import numpy as np
from tqdm import tqdm


@dataclass
class Tags:
    """Dataclass."""

    tags: Counter = field(default_factory=Counter)
    daily_tags: dict = field(default_factory=dict)
    days: tuple = ("2020-01-01", "2021-10-01")
    cooccurrence: Counter = field(default_factory=Counter)

    keywords: set = field(default_factory=set)

    def __post_init__(self):
        with open("data/keys-covid.txt", "rt") as fin:
            self.keywords = {k.strip() for k in fin} | {""}

    def __strip_hts__(self, tags):
        for t in set(tags.split("|")) - self.keywords:
            if len(t) == 0:
                continue
            if t.startswith("covid") and t.endswith("19"):
                continue
            yield t

    def update(self, day, tags, count=1):
        """Update with new tags."""
        tags = sorted(self.__strip_hts__(tags))
        if len(tags) <= 0:
            return
        for ht1, ht2 in combinations(tags, 2):
            self.cooccurrence[(day, ht1, ht2)] += count
        self.tags.update(tags)
        self.daily_tags.setdefault(day, Counter()).update(tags)

    def most_common(self, daily_num=20, tot_num=200):
        """Return the most common hashtags.

        daily_num set of `daily_num` most commons per each day
        tot_num set of `tot_num` most common globally.
        """
        daily = {t for tags in self.daily_tags.values() for t, _ in tags.most_common(daily_num)}
        tot = {tag for tag, _ in self.tags.most_common(tot_num)}
        return daily | tot

    def tensor(self, daily_num=20, tot_num=200) -> np.ndarray:
        """Return the tensor."""
        if len(self.days) == 2:
            days = {
                d.isoformat(): id
                for id, d in enumerate(cov_utils.daterange(self.days[0], self.days[1]))
            }
        else:
            days = {t: tid for tid, t in enumerate(sorted(self.daily_tags.keys()))}

        tags = self.most_common(daily_num=daily_num, tot_num=tot_num)
        tags = {t: it for it, t in enumerate(sorted(tags))}
        print(f"Using {len(tags)} HTs")
        tensor = np.zeros((len(tags), len(tags), len(days)), dtype=int)

        for (day, ht1, ht2), count in self.cooccurrence.items():
            if ht1 not in tags or ht2 not in tags:
                continue
            tensor[tags[ht1], tags[ht2], days[day]] += count
            tensor[tags[ht2], tags[ht1], days[day]] += count

        print(tensor.shape, tensor.sum())
        return tensor, days, tags

    def __len__(self):
        return len(self.tags)


def load_month(month, userset: set = None, use: str = "time"):
    """Load tweets of a month.

    Parameters
    ----------
    month : str
        month
    userset : set
        set of user ids to limit the retrieval
    use : str
        time: return day of tweet and retweet
        user: return the user id of tweet or retweet

    Return
    ------
    tweets : dict
        dictionary of ids: tuple(hashtags, day/userid, text)
    retweets : Counter
        counter of (tweeted_id, day/userid)
    """
    if use == "time":

        def info(x):
            return x["created_at"][:10]

    elif use == "user":

        def info(x):
            return x["from_user_id"]

    fname = pathlib.Path(f"../data/african_tweets_{month}.csv.gz")
    if not fname.is_file():
        return {}, Counter()
    with gzip.open(fname, "rt") as fin:
        reader = csv.DictReader(fin)
        if userset is None:
            tweets = {
                t["id"]: (t["hashtags"], info(t), t["text"]) for t in reader if "|" in t["hashtags"]
            }
        else:
            tweets = {
                t["id"]: (t["hashtags"], info(t), t["text"])
                for t in reader
                if "|" in t["hashtags"] and t["from_user_id"] in userset
            }
    with gzip.open(f"../data/african_retweets_{month}.csv.gz", "rt") as fin:
        reader = csv.DictReader(fin)
        if userset is None:
            retweeted = Counter([(rt["retweeted_id"], info(rt)) for rt in reader])
        else:
            retweeted = Counter(
                [(rt["retweeted_id"], info(rt)) for rt in reader if rt["from_user_id"] in userset]
            )

    return tweets, retweeted


def __monthlist__(typename) -> list:
    if "early" in typename:
        months = [f"2020-{x + 1:02d}" for x in range(7)]
        # months = [f"2020-{x + 1:02d}" for x in range(1)]
    elif "late" in typename:
        months = [f"2020-{x + 1:02d}" for x in range(7, 12)] + [
            f"2021-{x + 1:02d}" for x in range(9)
        ]
    else:
        months = [f"2020-{x + 1:02d}" for x in range(12)] + [f"2021-{x + 1:02d}" for x in range(9)]

    return months


def __get_users__(typename) -> dict:
    if typename == "nofr":
        with gzip.open("../data/african_users_extended.csv.gz", "rt") as fin:
            reader = csv.DictReader(fin)
            users = {
                user["from_user_id"]: user["geo_coding_extened"]
                for user in reader
                if len(user["geo_coding_extened"]) == 2 and user["geo_coding_extened"] != "FR"
            }
    elif "geo" in typename:
        with gzip.open("../data/african_users_extended.csv.gz", "rt") as fin:
            reader = csv.DictReader(fin)
            users = {
                user["from_user_id"]: user["geo_coding_extened"]
                for user in reader
                if len(user["geo_coding_extened"]) == 2
            }
    else:
        users = {}

    return users


def load_retweet_count(typename: str = "") -> Tags:
    """Return a dictionary with a retweet count per tweet per day."""
    months = __monthlist__(typename)

    # this will be (day, ht1, ht2)
    tensor = Tags(days=())
    # this is to remember tweets to hts
    full_tweets = {}
    users = __get_users__(typename)

    if "hcq" in typename:
        keys = cov_utils.get_keywords("hydroxychloroquine", get_minimal=True)["short"]
    else:
        keys = None

    def checktwt(hashtags, tweet, keys):
        for htg in keys:
            if htg in hashtags or htg in tweet:
                return True
        return False

    for month in tqdm(months):
        if "geo" in typename:
            tweets, retweets = load_month(month, userset=set(users.keys()), use="user")
        elif typename == "nofr":
            tweets, retweets = load_month(month, userset=set(users.keys()), use="time")
        else:
            tweets, retweets = load_month(month, userset=None, use="time")

        if "hcq" in typename:
            tweets = {tid: t for tid, t in tweets.items() if checktwt(t[0], t[2], keys)}

        full_tweets.update({tid: t[0] for tid, t in tweets.items()})

        if "geo" in typename:
            for hts, uid, text in tweets.values():
                tensor.update(users[uid], hts)
            for (tid, uid), count in retweets.items():
                if tid in full_tweets:
                    tensor.update(users[uid], full_tweets[tid], count=count)
        else:
            for hts, day, text in tweets.values():
                tensor.update(day, hts)
            for (tid, day), count in retweets.items():
                if tid in full_tweets:
                    tensor.update(day, full_tweets[tid], count=count)

    print("tag num", len(tensor))
    print("Cooccurence num", len(tensor.cooccurrence))
    return tensor


def main(typename: str = "full"):
    """Do the main."""
    basename = "../data/ht_cooccurrence_tensor"
    tags = load_retweet_count(typename=typename)

    tensor, days, tags = tags.tensor(daily_num=10, tot_num=100)

    with open(f"{basename}_{typename}.json", "wt") as fout:
        json.dump({"days": days, "hashtags": tags}, fout)
    np.savez_compressed(f"{basename}_{typename}.npz", cooccurrence=tensor)


if __name__ == "__main__":
    for typename in ["geo_early", "geo_late", "nofr", "full", "hcq"]:
        print("computing", typename)
        main(typename)
