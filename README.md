# exploring-twitter-api

Fetching tweets through the Twitter API v2 using Python. Inspired by [guide from Twitter](https://github.com/twitterdev/getting-started-with-the-twitter-api-v2-for-academic-research).

Requires `twarc` and `json` libraries. Paste your token in the `bearer_token` file.

Run a query using `python -m fetch_tweets query`, where `query` is replaced by an actual query - the tweets are stored in a file named after current time in UTC.