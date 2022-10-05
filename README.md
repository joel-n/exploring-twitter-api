# exploring-twitter-api

Fetching tweets through the Twitter API v2 using Python. Inspired by [guide from Twitter](https://github.com/twitterdev/getting-started-with-the-twitter-api-v2-for-academic-research).

Paste your API token in the `bearer_token` file. Requires `twarc` and `json` libraries for basic tweet processing. `requirements.txt` lists all the libraries used in the repo.

Run a query from the command line using `python -m fetch_tweets query`, where `query` is replaced by an actual query - the tweets are stored in a file named after current time in UTC.

Sample from the stream endpoint via the command line by running `python -m sample time`, where `time` is the duration (in seconds) for which to sample.

Retrieve full conversations (including replies, retweets and quote tweets) of the sample tweets using `retrieve_interactions_from_sample()` of the `miner` class. Make sure to provide a file `bearer_token.txt` or `academic_bearer_token.txt` with the API token. The `miner` class also supports retrieving the followers of a set of users.