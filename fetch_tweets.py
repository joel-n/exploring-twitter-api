from twarc import Twarc2, expansions
import datetime
import json
import sys


with open('tokens/bearer_token', 'r') as file:
    BEARER_TOKEN = file.read()

client = Twarc2(bearer_token=BEARER_TOKEN)

def append_file(file_name, result):

    with open(file_name, 'a+') as filehandle:
            for tweet in result:
                filehandle.write('%s\n' % json.dumps(tweet))


def main():

    """
    Collect tweets from a time period (in recent 7 days) through a query,
    or specify start and end date and time in UTC.
    """

    # start_time = datetime.datetime(2022, 1, 14, 0, 0, 0, 0, datetime.timezone.utc)
    # end_time = datetime.datetime(2022, 1, 15, 0, 0, 0, 0, datetime.timezone.utc)
    # end_time = datetime.datetime.utcnow()

    args = sys.argv[1:]
    query = ' '.join(args)

    # Use recent search endpoint
    search_results = client.search_recent(query=query, max_results=10)
    # search_results = client.search_recent(query=query, start_time=start_time, end_time=end_time, max_results=10)

    # Examples of creating user timelines, finding mentions, followers/following etc.
    # search_results = client.timeline(user="twitterdev")
    # search_results = client.mentions(user="twitterdev")
    # search_results = client.following(user="twitterdev")
    # search_results = client.followers(user="twitterdev")

    t_query = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'{t_query}.txt'
    
    # All Tweets that satisfy the query are returned in pages
    for page in search_results:
        result = expansions.flatten(page)
        append_file(file_name, result)


if __name__ == '__main__':
    main()