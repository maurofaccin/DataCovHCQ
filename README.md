# Hydroxychloroquine international controversy in francophone Titter

In this repository we provide dataset used and produced in:

> **XXX**
> *Faccin, Emilien, Gargiulo*
> [PLOS ONE, 17(8) p.1-19 (2022)](https://doi.org/10.1371/journal.pone.0271157).

The cited article contains a detailed description of the datasets' construction processes.
If you use this in your project, please add a reference to the above article.

## Datasets

### Tweet IDs

The file `data/african_tweets_YYYY-MM.csv.gz` contains the list of tweets and retweets IDs used in the above publication, split by month.
The file `./data/african_users_extended.csv.gz` contains the list of users IDs as well as their country (in the `geo_coding_extended` column one can find the full geographic tagging used).


### Twitter APIs keywords

This dataset uses the same `keywords` as in [DataCovVac](https://github.com/maurofaccin/DataCovVac) for Twitter streaming and search APIs.
The dataset [`./data/keywords.json`](https://github.com/maurofaccin/DataCovVac/blob/main/data/keywords.json) is divided into three sets that corresponds to the three tweet datasets (DataVac, DataCov, DataHC).

The dataset format is as follows:

```json
{
    "DataVac": [...],
    "DataCov": [...],
    "DataHC": [...]
}
```

## Codes

In the `covhcq-code` folder we provide a number of python scripts that extract the data needed to plot.
The code requires tweets and retweets to be hydrated.
Files containing the tweets should be named `./data/african_tweets_YYYY-MM.csv.gz` and have the following columns:

```
id,time,created_at,text,from_user_name,from_user_id,to_user_id,quoted_id,quoted_user_id,mentioned_user_ids,location,links,hashtags
```

Files containing retweets should be named `./data/african_retweets_YYYY-MM.csv.gz` and have the following columns:

```
id,retweeted_id,time,from_user_id,created_at
```

**WARNING**: script are provided as is, and they require user intervention in order to update paths and possibly other data.

### Plotting

In the `covhcq-plots` folder one can find the python scripts used to reproduce the plots of the above paper.
