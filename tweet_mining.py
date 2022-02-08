import os
import sys
import json
import logging
import datetime
import threading
import numpy as np
import pandas as pd
import time as pytime
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

from twarc import Twarc2, expansions

with open('tokens/bearer_token.txt', 'r') as file:
    BEARER_TOKEN = file.read()

client = Twarc2(bearer_token=BEARER_TOKEN)

def print_dict(d: dict, indent: str) -> None:
    """Prints the keys of a dictionary to visualize the
    content of e.g., attributed of a returned tweet.
    
    Not all tweet objects contain the same attributed.
    
    """
    if isinstance(d, dict):
        for key in d:
            print(indent, key)
            print_dict(d[key], indent+'-')
    elif isinstance(d, list):
        for l in d:
            print(indent+' list: [')
            if isinstance(l, dict):
                print_dict(l, indent+'\t-')
            print(indent+'\t]')
    if indent == ' ':
        print(" ")
        
    return


def append_ids_to_file(file_name: str, result: list) -> None:
    """Append a file with the 'id' attribute of a list of objects.
    Use case is dehydration of tweets or users, e.g., write the
    tweetIDs of a list of tweets.
    
    Args:
        - file_name: name of the file to append
        - result: list of objects to append, retrieved
        from expansions.flatten(result_page)    
        
    No return value.
    
    """
    with open(file_name, 'a+') as filehandle:
        for element in result:
                filehandle.write('%s\n' % element['id'])
                
                
                
def append_objs_to_file(file_name: str, result: list) -> None:
    """Append a file with the objects in result.
    Use case: append a file with a flattened search
    query (tweets, users etc.).
    
    Args:
        - file_name: name of the file to append
        - result: list of objects to append, retrieved
        from expansions.flatten(result_page)
        
    No return value.
    """
    with open(file_name, 'a+') as filehandle:
            for obj in result:
                filehandle.write('%s\n' % json.dumps(obj))
                
                
def read_file(file_name: str) -> list[dict]:
    """Returns a list of the contents in a .jsonl-file.
    
    Args:
        - file_name: name of the file to read
        
    Returns:
        - objs: list of json objects in the file
    
    """
    objs = []
    with open(file_name, 'r') as filehandle:
            for obj in filehandle:
                objs.append(json.loads(obj))
    return objs
    # yield objs



def create_scatter_plot(x, y, path: str, title='', xlab='', ylab='') -> None:
    """Plot data x and y in a scatter plot and save
    to the provided path.
    
    Args:
        - x: data for x axis (list or array)
        - y: data for y axis (list or array)
        - path: path to save figure to
        - title: figure title
        - xlab: x-axis label
        - ylab: y-axis label
        
    No return value.    
    """
    plt.figure(figsize=(8,8), clear=True)
    plt.plot(x, y, 'o')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(path)
    return


def create_hist(x, bins, path: str, title='', xlab='', ylab='') -> None:
    """Plot data x and y in a histogram and save
    to the provided path.
    
    Args:
        - x: data for x axis (list or array)
        - bins: integer or list/array that specifies the number of
        bins to use or the bins limits.
        - path: path to save figure to
        - title: figure title
        - xlab: x-axis label
        - ylab: y-axis label
        
    No return value.    
    """
    plt.figure(figsize=(8,8), clear=True)
    plt.hist(x, bins)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(path)
    return

def recent_search(query_: str, max_results_pp=100) -> list[dict]:
    """Query the Twitter API for data using
    the recent_search endpoint.
    
    Args:
        - query_: query string (max 512 characters for essential
        or elevated access, 1024 for acadeic access).
        - max_results_pp: the maximum results to return per page,
        note that all matching results will be returned
        
    Returns:
        - tweets: list of tweets in the form of json objects
    
    """
    search_results = client.search_recent(query=query_, max_results=max_results_pp)
    tweets = []

    for page in search_results:
        result = expansions.flatten(page)
        for tweet in result:
            tweets.append(tweet)
    return tweets


def counts(query_: str) -> list[dict]:
    """Query the counts endpoint for the volume of tweets
    returned by a query. Dates need to be set appropriately
    or else no results will be returned (7-day limit).
    
    Args:
        - query_: query string (max 512 characters for essential
        or elevated access, 1024 for acadeic access).
        
    Returns:
        - volume: list of tweet volumes over the time spans
    
    """
    volume = []
    start_time = datetime.datetime(2022, 1, 14, 0, 0, 0, 0, datetime.timezone.utc)
    end_time = datetime.datetime(2022, 1, 15, 0, 0, 0, 0, datetime.timezone.utc)
    count_results = client.counts_recent(query=query_, start_time=start_time, end_time=end_time)
    for page in count_results:
        volume.append(page['data'])
        #print(json.dumps(page['data']))
        #break
    return volume


def get_retweeters(tweet_id: str) -> list[str]:
    """Query the APi for retweeters of a specific tweet.
    
    Args:
        - tweet_id: ID of the tweet for which to fetch retweeters
        
    Returns:
        - retweeters: list of user IDs that retweeted tweet_id
    """
    retweeters = []
    results = client.retweeted_by(tweet_id)
    for page in results:
        res = expansions.flatten(page)
        for retweeter in res:
            retweeters.append(retweeter['id'])
    return retweeters


def retrieve_conversations_recent(sample_file: str) -> None:
    """Retrieves and saves conversations that contain
    the tweets in a sampled file. The files are saved
    in the sampled_conversations folder. Uses recent
    search, not full archive queries.
    
    Args:
        - sample_file: path to the sampled tweets. The file should be
        the output of the sampled stream in .jsonl format, with the
        first attribute 'data' for each line (default when sampling
        from the command line).
        
    No return value.
    """
    
    sample = read_file(sample_file)
    print('Retrieving conversations for {} tweets.\
    With 1200 queries per hour this should take {:.3f} minutes ({:.3f} hours)'.format(len(sample), len(sample)/30, len(sample)/1800))
    t1 = pytime.time()
    tot_conv_retrieved = 0
    
    for i,t in enumerate(sample):
        conv_id = t['data']['conversation_id']
        conv_query = f'conversation_id:{conv_id}'
        
        # Calling recent search (rate limit: 450 requests/15 minutes)
        conv_res = client.search_recent(query=conv_query)

        filename = f'sampled_conversations/{conv_id}_conversation-tweets_{i}.jsonl'
        pages = 0
        results = 0
        for page in conv_res:
            pages += 1
            result = expansions.flatten(page)
            results += len(result)
            with open(filename, 'a+') as filehandle:
                for tweet in result:
                    filehandle.write('%s\n' % json.dumps(tweet))
        logging.info(f'conversation {conv_id} resulted in {pages} pages and a total of {results} results.')
        if pages > 0:
            tot_conv_retrieved += 1
    
    t2 = pytime.time() - t1
    logging.info(f'Conversation retrieval took {t2} seconds. Out of {len(sample)} conversations, {tot_conv_retrieved} contained replies.')
    print('Finished retrieving {} conversations in {:.3f} minutes ({:.3f} hours)'.format(tot_conv_retrieved, t2/60, t2/3600))
    
    return


def get_root_tweets(conv_ids: list[str]) -> list[dict]:
    """Returns a list of tuples with information on a
    list of tweets (tweetIDs). Use case is to retrieve
    the id of the root of a conversation in order to query
    for retweets.
    
    Args:
        - conv_ids: list of tweet or conversation IDs
        
    Returns:
        - roots: list of tuples containing the author ID, engagement
        metrics, tweet text and time stamp of the original tweet.
    
    """
    roots = []
    results = client.tweet_lookup(conv_ids)
    for page in results:
        res = expansions.flatten(page)
        for orig_tweet in res:
            cid = orig_tweet['conversation_id']
            append_objs_to_file(f'root_tweets/{cid}_root.jsonl', [orig_tweet])
            
            roots.append({'id':         orig_tweet['author_id'],
                          'n_retweets': orig_tweet['public_metrics']['retweet_count'],
                          'text':       orig_tweet['text'],
                          'created_at': orig_tweet['created_at']})
    return roots


def get_saved_conversation_ids(lower_bound: int, upper_bound: int) -> list:
    """Returns a list of conversation ids between a given
    lower and upper bound from the folder of saved conversations.
    
    Args:
        - lower_bound: integer lower limit of conversation IDs to include
        - upper_bound: integer upper limit of conversation IDs to include
    
    Returns:
        - conv_ids: a list of conversation IDs from the folder
    
    """
    
    file_paths = os.listdir('sampled_conversations')
    conv_ids = []
    
    for conv_file_name in file_paths:
        conv_id = conv_file_name.split('_')[0]
        if int(conv_id) >= lower_bound and int(conv_id) <= upper_bound:
            conv_ids.append(conv_id)

    return conv_ids


def get_retweets_of(conv_ids: list[str]) -> None:
    """Get retweets of conversation roots and stores them in a .jsonl-file.
        
    Args:
        - conv_ids: list of conversation IDs to fetch retweets for
        
    No return value. Saves the retweets in a .jsonl file named
    after the conversation ID in the retweets folder. Will create
    or append files equal to the number elements in the supplied
    conv_id list provided that results are returned from the API.
    
    """
    
    # Get list of dicts containing author ID, retweet
    # count, text and time stamps for original tweets
    roots = get_root_tweets(conv_ids)
    
    n_retweets = 0
    for r in roots:
        n_retweets += r['n_retweets']
        
    print('Total of {} retweets.\
    With 300 queries per hour this should take {:.3f} minutes ({:.3f} hours)'.format(n_retweets, n_retweets/5, n_retweets/300))
    t1 = pytime.time()
    
    for i, root in enumerate(roots):
        retweets = []
        
        # Find retweeting authors
        retweeters = get_retweeters(conv_ids[i])
        for retweeter in retweeters:
            # Query for any retweet by author of original tweet author, containing the same text
            # Should only return 1 result in practice (if querying on text)
            root_author = root['id']
            text = root['text']
            #retweets = recent_search(query_=f'retweets_of:{root['id']} from:{retweeter} "{text}"')
            retweets_result = recent_search(query_=f'retweets_of:{root_author} from:{retweeter}')
            
            for rt in retweets_result:
                # Find relevant results: filter on referenced_tweets
                # Should only contain reference to 1 tweet if it is a pure retweet;
                # if so use rt['referenced_tweets'][0]
                for t in rt['referenced_tweets']:
                    if t['id'] == conv_ids[i]:
                        retweets.append(rt)
                        break
                # break if RT found to avoid duplicates?
                    
        # Add RTs to the conversation file
        append_objs_to_file(f'retweets/{conv_ids[i]}.jsonl', retweets)
        logging.info(f'Retrieved {len(retweets)} retweets of tweet {conv_ids[i]}.')
    
    t2 = pytime.time() - t1
    logging.info(f'Retweets retrieval took {t2} seconds.')
    print('Finished in {:.3f} minutes ({:.3f} hours)'.format(t2/60, t2/3600))
    
    return

def retweets_metrics(conv_ids: list[str]) -> None:
    """Computes retweet metrics and print to a file.
    Plots retweets over time from the initial post.
    Saves images to 'sampled_conversation_graphs' folder.
    
    TODO (suggestions): Save retweet times to files, and
    use this information in another function.
    
    Args:
        - conv_ids: list of conversation IDs (that have retweet
        files in the 'retweet' folder, and root files in the
        'root_tweet' folder).
        
    Metrics:
    - Retweets over time; histogram/pdf/cdf (per root)
    - Number of retweets; histogram (for all roots)
    - Retweets vs root followers; scatter (for all roots)
    - Final engagement time; histogram (for all roots)
    - Final engagement time v. followers; scatter (for all roots)
    - Final engagement time v. retweets; scatter (for all roots)
        
    No return value.
    """
    
    n_retweets = []
    root_followers = []
    final_rt_time = []
    
    for conv_id in conv_ids:
        root = read_file(f'root_tweets/{conv_id}_root.jsonl')[0]
        retweets = read_file(f'retweets/{conv_id}.jsonl')

        n_retweets.append(len(retweets))
        root_followers.append(root['author']['public_metrics']['followers_count'])
        
        engagement_time = []   
        t0 = datetime.datetime.strptime(root['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')
        for rt in retweets:
            time = datetime.datetime.strptime(rt['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')
            dt = time-t0
            engagement_time.append((86400*dt.days + dt.seconds)/3600)

        engagement_time = np.sort(engagement_time)
        
        if(len(engagement_time)) > 0:
            final_rt_time.append(engagement_time[-1])
        else:
            final_rt_time.append(0)
        #for i in range(len(engagement_time)):
        #    engagement_time[i] = engagement_time[i] - engagement_time[-1]
        
        create_hist(engagement_time, bins=50, path='sampled_conversations_graphs/{conv_id}_retweets.svg',
                title='Engagement times', xlab='time (h)', ylab='counts')

    create_scatter_plot(n_retweets, root_followers, 'sampled_conversations_graphs/retweets_vs_followers.svg',
                        title='Retweets as a function of followers', xlab='followers', ylab='retweets')
    create_scatter_plot(final_rt_time, root_followers, 'sampled_conversations_graphs/followers_vs_final_time.svg',
                        title='Final engagement time', xlab='final retweet time (h)', ylab='number of followers')
    create_scatter_plot(final_rt_time, n_retweets, 'sampled_conversations_graphs/retweets_vs_final_time.svg',
                        title='Final engagement time', xlab='final retweet time (h)', ylab='number of retweets')
    create_hist(final_rt_time, bins=50, path='sampled_conversations_graphs/final_time_distribution.svg',
                title='Final engagement time distribution', xlab='final retweet time (h)', ylab='counts')
    create_hist(n_retweets, bins=50, path='sampled_conversations_graphs/retweets_distribution.svg',
                title='Retweets distribution', xlab='retweets', ylab='counts')
    
    d = {'conv_id':conv_ids,
         'n_retweets':n_retweets,
         'root_followers':root_followers,
         'final_rt_time':final_rt_time}
    
    pd.DataFrame(data=d).to_csv('rt_data.csv',index=False)
    
    return


def get_conversation_dict(conv_tweets: list[dict]) -> tuple(dict, list):
    """Returns a dictionary containing information on
    the tweets in the conversation along with a list
    of their engagement times in hours after tweet zero.
    Dictionaries in Python >3.7 are ordered.
    
    Args:
        - conv_tweets: a list with tweet .json objects
        
    Returns:
        - conv_dict: a dictionary mapping tweet ids to tweet info
        - engagement_time: list of times in hours
    
    """
    conv_dict = {}
    time_stamps = []
    for tw in conv_tweets:
        ref_id = None
        ref_auth_id = None
        try:
            for ref_tw in tw['referenced_tweets']:
                if ref_tw['type'] == 'replied_to':
                    ref_id = ref_tw['id']
                    ref_auth_id = ref_tw['author_id']
                    break
        except Exception as e:
            logging.warning(e)

        conv_dict[tw['id']] = {'author_id':      tw['author_id'],
                               'author_name':    tw['author']['name'],
                               'time_stamp':     tw['created_at'],
                               'public_metrics': tw['public_metrics'],
                               'referenced_id':  ref_id,
                               'referenced_author_id': ref_auth_id}
                             # 'text_snippet':tw['text'][0:4],
        
        time_stamps.append(tw['created_at'])
        
    engagement_time = []
    t0 = datetime.datetime.strptime(time_stamps[0], '%Y-%m-%dT%H:%M:%S.000Z')
    for ts in time_stamps:
        time = datetime.datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.000Z')
        dt = time-t0
        engagement_time.append((86400*dt.days + dt.seconds)/3600)

    for i in range(len(engagement_time)):
        engagement_time[i] = engagement_time[i] - engagement_time[-1]
        
    return conv_dict, engagement_time

def create_conversation_network(conv_dict, engagement_time):
    """Returns a directed networkx graph of the conversation network
    where the user participating in the conversation are the nodes.
    
    Args:
        - conv_dict: a conversation dictionary, the
        output of get_conversation_dict().
    
    Returns:
        - DiG: a directed graph based on the conversation dictionary
        The nodes have author and time as an attribute.
    """
    DiG = nx.DiGraph()
    for i, key in enumerate(conv_dict):
        DiG.add_node(conv_dict[key]['author_id'], author=conv_dict[key]['author_name'], time=engagement_time[i],
                     rts=conv_dict[key]['public_metrics']['retweet_count'],
                     likes=conv_dict[key]['public_metrics']['like_count'])

    for key in conv_dict:
        if conv_dict[key]['referenced_author_id'] is not None:
            DiG.add_edge(conv_dict[key]['author_id'], conv_dict[key]['referenced_author_id'])
        
        
    """Fix the problem with some tweets getting no time:
    
    # e.g. by 
    for n in nx.nodes(DiG):
        try:
            print(DiG.nodes[n]['time'])
        except:
            print(f'no time for {n}')
            DiG.nodes[n]['time'] = 0
            print("new time", DiG.nodes[n]['time'])

    """      
    return DiG

def create_conversation_network_tree(conv_dict, engagement_time):
    """Returns a directed networkx graph of the conversation network
    where the messages are nodes.
    
    Args:
        - conv_dict: a conversation dictionary, the
        output of get_conversation_dict().
    
    Returns:
        - DiG: a directed graph based on the conversation dictionary
        The nodes have author and time as an attribute.
    """
    DiG = nx.DiGraph()
    for i, key in enumerate(conv_dict):
        DiG.add_node(key, author=conv_dict[key]['author_name'], time=engagement_time[i],
                     rts=conv_dict[key]['public_metrics']['retweet_count'],
                     likes=conv_dict[key]['public_metrics']['like_count'])

    for key in conv_dict:
        DiG.add_edge(key, conv_dict[key]['referenced_id'])
        
        
    """Fix the problem with some tweets getting no time:
    
    # e.g. by 
    for n in nx.nodes(DiG):
        try:
            print(DiG.nodes[n]['time'])
        except:
            print(f'no time for {n}')
            DiG.nodes[n]['time'] = 0
            print("new time", DiG.nodes[n]['time'])

    """    
        
        
    return DiG


def assign_time_attributes(graph):
    times = []
    for n in nx.nodes(graph):
        try:
            times.append(graph.nodes[n]['time'])
        except:
            graph.nodes[n]['time'] = 0
            times.append(graph.nodes[n]['time'])
    return times



def plot_engagement(engagement_time, save_path=None):
    max_ = max(engagement_time)
    n_bins = max(10, int(max_*6)) # make one bin per 10 minutes
    
    plt.figure(figsize=(9,9))
    n, bs, _ = plt.hist(engagement_time, bins=n_bins)
    plt.title('Replies after posting')
    plt.xlabel('hours')
    plt.ylabel('replies')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    n_norm = n/np.sum(n)
    cdf = np.cumsum(n_norm)
    cdf = np.concatenate((np.zeros(1), cdf)) # same length as bins
    plt.figure(figsize=(9,9))
    plt.plot(bs, cdf, '-')
    plt.title('Engagement CDF')
    plt.xlabel('hours')
    if save_path:
        plt.savefig(f'{save_path[:-4]}_cdf.svg')
    else:
        plt.show()
    
    return

def get_users(username, user_fields_='created_at,id,name,username,public_metrics'):
    """Returns a hydrated list of user profiles
    from the list of IDs provided.
    
    Args:
        - username: list of usernames for the accounts to look up.
        - user_fields_: optional string with query attributes
        
    Returns:
        - userlist: list of user profiles with the attributes
        in user_fields_.
    """
    
    users = client.user_lookup(username, usernames=True, expansions=None, tweet_fields=None, user_fields=user_fields_)
    userlist = []
    for page in users:
        result = expansions.flatten(page)
        for usr in result:
            userlist.append(usr)
    return userlist


def get_followers(user_, load=True):
    """Saves and (optionally) returns a list of user profiles and IDs
    of the accounts following a user. The profiles are stored in the
    follower_ids folder.
    
    Args:
        - user: ID of the user for which to retrieve followers
        - load: indicates whether the list of user profiles
        should be returned as a variable. If False and empty
        list will be returned.
        
    Returns:
        - flws: (OPTIONAL) a list of user profiles. Will be empty
        if 'load' is set to be false.
    """
    followers = client.followers(user=user_, user_fields='created_at,id,name,username,public_metrics')
    t_query = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'{user_}_followers_{t_query}.jsonl'
    id_file_name = f'follower_ids/{user_}_follower-ids_{t_query}.txt'
    
    flws = []
    for page in followers:
        result = expansions.flatten(page)
        append_objs_to_file(file_name, result)
        append_ids_to_file(id_file_name, result)
        if load:
            for usr in result:
                flws.append(usr)
    
    return flws


def get_following(user_, load=True):
    """Saves and (optionally) returns a list of user profiles and IDs
    of the accounts that a user follows. The profiles are stored in the
    following_ids folder.
    
    Args:
        - user: ID of the user for which to retrieve followings
        - load: indicates whether the list of user profiles
        should be returned as a variable. If False and empty
        list will be returned.
        
    Returns:
        - flwing_ids: (OPTIONAL) a list of user profiles. Will be empty
        if 'load' is set to be false.
    """
    
    following = client.following(user=user_, user_fields='created_at,id,name,username,public_metrics')
    t_query = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'{user_}_following_{t_query}.jsonl'
    id_file_name = f'following_ids/{user_}_following-ids_{t_query}.txt'
    
    flwing_ids = []
    for page in following:
        result = expansions.flatten(page)
        append_objs_to_file(file_name, result)
        append_ids_to_file(id_file_name, result)
        if load:
            for usr in result:
                flwing_ids.append(usr['id'])
                 
    return flwing_ids


def get_downloaded_user_followers(folder_path: str):
    """Return a list of the user IDs of which we
    have retrieved followers.

    Args:
        - folder_path: path to the files of followers

    Returns:
        - user_ids: list of user IDs for which we have
        retrieved followers for.
    
    """
    file_paths = os.listdir(folder_path)
    user_ids = []
    for filename in file_paths:
        user_id = filename.split('_')[0]
        user_ids.append(user_id)
    return user_ids


def extend_database(folder_path: str, already_retrieved_ids: set, max_retrievals=1e9) -> set:
    """Extends the database of followers.
    Not particularly useful due to the rate
    limits on follower retrieval.

    Args:
        - folder_path: path to the follower files
        - already_retrieved_ids: set of user IDs
        for which followers have been retrieved
        - max_retrievals: maximum number of queries
        to send to the API

    Returns:
        - already_retrieved_ids: updated set of
        user IDs for which followers have been fetched
    """

    query_count = 0 # Rate limit is 15 queries
    exit = False
    
    file_paths = os.listdir(folder_path)
    for filename in file_paths:
        with open(f'{folder_path}/{filename}', mode='r', encoding="utf8") as file:
            user_ids = file.readlines()
                
        for line in user_ids:
            id_ = line[:-1] # Remove '\n'
            if id_ not in already_retrieved_ids:
                print(f'retrieving followers of user id {id_}')
                already_retrieved_ids.add(id_)
                try:
                    get_followers(id_, load=False)
                except Exception as e:
                    logging.warning(e)
                    print(f'retrieving followers of user id {id_} failed')
                query_count += 1
            else:
                print(f'user id {id_} already retrieved')
                
            if query_count % 15 == 0:
                print(f'query {query_count}, sleeping for 15 minutes...', end=' ')
                pytime.sleep(900) # Rest for 15 minutes
                print('done')
            if query_count > max_retrievals:
                print(f'reached {query_count} queries, exiting')
                exit = True
                break
            
        if exit:
            break
    
    return already_retrieved_ids     


def build_partial_network(gexf_name, followers=True, following=True):
    """Create a .gexf network for the followers and followings of
    the set of users contained in the follower_ids and following_id
    folders.
    
    Args:
        - gexf_name: name of the output file (excluding '.gexf')
        
    No return value.
    """
    dir_graph = nx.DiGraph()
    
    # Get followers
    if followers:
        file_paths = os.listdir('follower_ids')
        for filename in file_paths:

            dest_id = filename.split('_')[0]

            with open(f'follower_ids/{filename}', mode='r', encoding="utf8") as file:
                user_ids = file.readlines()

            for line in user_ids:
                id_ = line[:-1] # Remove '\n'
                dir_graph.add_edge(id_, dest_id)
                
    if following:
        file_paths = os.listdir('following_ids')
        for filename in file_paths:

            source_id = filename.split('_')[0]

            with open(f'following_ids/{filename}', mode='r', encoding="utf8") as file:
                user_ids = file.readlines()

            for line in user_ids:
                id_ = line[:-1] # Remove '\n'
                dir_graph.add_edge(source_id, id_)
                
            
    nx.readwrite.gexf.write_gexf(dir_graph, path=f'{gexf_name}.gexf', encoding='utf-8')
    return


def create_conversation_graphs(dir_, show_plot=False, plot_with_colors=True):
    """Create conversation graphs in .svg and .gexf (Gephi) formats.
    Also creates histograms of the engagement in replies over time.
    Graphs are saved in 'sampled_conversations_graphs' folder.
    
    The kamada kawai layout seems to be expensive to compute for large graphs.
    One may just as well use the -gexf-file only, but the images cannot be exported
    to .svg format.
    
    Args:
        - dir_: directory consisting of conversation files
        (.jsonl formatted files containing tweets in the conversation)
        
    No return value.
    
    """
    file_paths = os.listdir(dir_)

    for conv_file_name in file_paths:

        conv_id = conv_file_name.split('_')[0]
        graph_file_name = f'sampled_conversations_graphs/{conv_id}_conversation_graph_users.gexf'

        conv_sample = read_file(f'sampled_conversations/{conv_file_name}')

        conv_dict, engagement_time = get_conversation_dict(conv_sample)

        dir_graph_sample = create_conversation_network(conv_dict, engagement_time)
        
        """
        # Room for computing metrics on the graphs!
        """
        
        nx.readwrite.gexf.write_gexf(dir_graph_sample, path=graph_file_name, encoding='utf-8')

        if plot_with_colors:
            assign_time_attributes(dir_graph_sample)

            plt.figure(figsize=(13,13))
            vmax_ = int(engagement_time[-1]+1)
            nc = [v for k,v in nx.get_node_attributes(dir_graph_sample, 'time').items()]
            nx.draw_kamada_kawai(dir_graph_sample, with_labels=False, font_weight='bold', node_color = nc, vmin=0, vmax=vmax_, cmap = plt.cm.get_cmap('rainbow'))
            plt.savefig(f'sampled_conversations_graphs/{conv_id}_conversation_graph_users.svg')
            if show_plot:
                plt.show()
        else:
            plt.figure(figsize=(13,13))
            nx.draw_kamada_kawai(dir_graph_sample, with_labels=False)
            plt.savefig(f'sampled_conversations_graphs/{conv_id}_conversation_graph_users.svg')
            if show_plot:
                plt.show()
        
        if show_plot:
            plot_engagement(engagement_time)
        else:
            plot_engagement(engagement_time, save_path=f'sampled_conversations_graphs/{conv_id}_engagement_time.svg')
        
        
        
def sample_num_retweets(dir_):
    """Collects the number of retweets of the original tweet
    in the conversation files located in dir_.
    
    Args:
        - dir_: directory consisting of conversation files
        (.jsonl formatted files containing tweets in the conversation)
        
    Returns:
        - RTs: A list of the number of retweets of the conversation root.
        - cid: A list corresponding to the conversation id to each number
        in RTs. (Not every conversation has an accessible root.)
    """
    
    file_paths = os.listdir(dir_)
    
    RTs = []
    cid = []
    for conv_file_name in file_paths:
        conv_id = conv_file_name.split('_')[0]
        with open(f'sampled_conversations/{conv_file_name}', 'r') as filehandle:
            for line in filehandle:
                obj = json.loads(line)
                try:
                    found = False
                    for ref_tw in obj['referenced_tweets']: # check referenced tweets
                        # check that the referenced tweet is indeed the root
                        if ref_tw['id'] == conv_id: # and ref_tw['type'] == 'replied_to' ?
                            RTs.append(ref_tw['public_metrics']['retweet_count'])
                            cid.append(conv_id)
                            found = True
                            break
                    if found: # go to next file
                        break
                except:
                    continue
                
    return RTs, cid

"""
# Event controlled sampling
thread_stop = threading.Event()
thread_stop.clear()

t_query = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
file_name = f'sampled_tweets_{t_query}.jsonl'

def sample():
    append_objs_to_file(file_name, client.sample(event=thread_stop))

# Begin streaming tweets
sample_thread = threading.Thread(target=sample)
sample_thread.daemon = True
sample_thread.start()

# Stop sampling
thread_stop.set()

"""