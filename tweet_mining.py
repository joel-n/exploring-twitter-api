import os
import re
import json
import logging
import datetime
import threading # for sampling from stream
import numpy as np
import pandas as pd
import time as pytime
import networkx as nx
from file_manip import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from twarc import Twarc2, expansions
# import seaborn as sns # plotting tools


academic_access = True

if academic_access:
    with open('tokens/academic_bearer_token.txt', 'r') as file:
        BEARER_TOKEN = file.read()
else:
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


def create_plot(x, y, path: str, format_='-', title='', xlab='', ylab='') -> None:
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
    if y is None:
        plt.plot(x, format_, color='dodgerblue')
    else:
        plt.plot(x, y, format_, color='dodgerblue')
    plt.title(title, size=24)
    plt.xlabel(xlab, size=18)
    plt.ylabel(ylab, size=18)
    plt.savefig(path)
    plt.close('all')
    return


def create_hist(x, bins, path: str, title='', xlab='', ylab='',
                log_=False, overlay_line=False, t=None, y=None) -> None:
    """Plot data x and y in a histogram and save
    to the provided path. Optionally plots a line
    over the histogram.
    
    Args:
        - x: data for x axis (list or array)
        - bins: integer or list/array that specifies the number of
        bins to use or the bins limits.
        - path: path to save figure to
        - title: figure title
        - xlab: x-axis label
        - ylab: y-axis label
        - overlay_line: plots line points (t,y) over histogram
        - t: x-values of overlaid line
        - y: y-values of overlaid line
        
    Returns:
        - n: values of the histogram bins
        - bs: bins generated or given
    """
    plt.figure(figsize=(8,8), clear=True)
    n, bs, _ = plt.hist(x, bins, color='palegreen', log=log_)
    plt.title(title, size=24)
    plt.xlabel(xlab, size=18)
    plt.ylabel(ylab, size=18)
    
    if overlay_line and t is not None and y is not None:
        plt.plot(t,y,'r')
    
    plt.savefig(path)
    plt.close('all')
    return n, bs


def create_loglog_hist(x, n_bins, path: str, title='', xlab='', ylab='') -> None:
    """Plot data x and y in a histogram in log-log scale
    and save to the provided path.
    
    Args:
        - x: data for x axis (list or array)
        - n_bins: integer that specifies the number of bins
        - path: path to save figure to
        - title: figure title
        - xlab: x-axis label
        - ylab: y-axis label
        
    No return value.    
    """
    
    bins_ = np.concatenate((np.zeros(1), np.logspace(0, np.log10(max(x) + 1), num=n_bins, endpoint=True, base=10.0, dtype=None, axis=0)))
    plt.figure(figsize=(8,8), clear=True)
    plt.hist(x, bins_, color='palegreen', log=True)
    plt.title(title, size=24)
    plt.xlabel(xlab+' (log scale)', size=18)
    plt.ylabel(ylab+' (log scale)', size=18)
    plt.xscale('log')
    plt.savefig(path)
    plt.close('all')
    return


def create_ccdf(data, path: str, title='', xlab='', ylab='', loglog=False) -> None:
    """Plot the complementary cumulative density
    function (optionally in log-log scale).
    
    Args:
        - data: data to plot
        - path: path to save figure to
        - title: figure title
        - xlab: x-axis label
        - ylab: y-axis label
        - loglog: boolean indicating whether
        the ccdf is plotted in loglog-scale
    
    """
    sorted_x = np.sort(data)
    n = len(data)
    ccdf = [1 - i/n for i in range(1, n+1)]
    
    plt.figure(figsize=(8,8), clear=True)
    plt.plot(sorted_x, ccdf, color='palegreen')
    plt.title(title, size=24)
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(xlab+' (log scale)', size=18)
        plt.ylabel(ylab+' (log scale)', size=18)
    else:
        plt.xlabel(xlab, size=18)
        plt.ylabel(ylab, size=18)
    plt.savefig(path)
    plt.close('all')
    return


def recent_search(query_: str, max_results_pp=100, tweet_fields_=None, user_fields_=None) -> list[dict]:
    """Query the Twitter API for data using
    the search/recent endpoint.
    
    Args:
        - query_: query string, max 512 characters
        - max_results_pp: the maximum results to return per page,
        note that all matching results will be returned
        
    Returns:
        - tweets: list of tweets in the form of json objects
    
    """
    search_results = client.search_recent(query=query_, max_results=max_results_pp,
                                          tweet_fields=tweet_fields_, user_fields=user_fields_)
    tweets = []

    for page in search_results:
        result = expansions.flatten(page)
        for tweet in result:
            tweets.append(tweet)
    return tweets


def full_archive_search(query_: str, max_results_pp=500, tweet_fields_=None, user_fields_=None,
                        media_fields_=None, poll_fields_=None, place_fields_=None, expansions_=None) -> list[dict]:
    """Query the Twitter API for data using
    the search/all endpoint.
    
    Args:
        - query_: query string, max 1024 characters
        - max_results_pp: the maximum results to return per page,
        note that all matching results will be returned
        
    Returns:
        - tweets: list of tweets in the form of json objects
    
    """    
    search_results = client.search_all(query=query_, max_results=max_results_pp, expansions=expansions_,
                                       tweet_fields=tweet_fields_, user_fields=user_fields_,
                                       media_fields=media_fields_, poll_fields=poll_fields_,
                                       place_fields=place_fields_)
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


def get_single_root(root_id: str) -> bool:
    """Retrieves a single (root) tweet defined by root_id.
    
    Args:
        - root_id: ID of tweet to retrieve
        
    Returns:
        - A boolean indicating whether the tweet has been
        retrieved.
    """
    if not os.path.isfile(f'root_tweets/{root_id}_root.jsonl'):
        result = client.tweet_lookup([root_id])
        for page in result:
            if not 'errors' in page:
                res = expansions.flatten(page)
                for orig_tweet in res:
                    cid = orig_tweet['conversation_id']
                    append_objs_to_file(f'root_tweets/{cid}_root.jsonl', [orig_tweet])

    return os.path.isfile(f'root_tweets/{root_id}_root.jsonl')


def retrieve_conversations(sample_file: str) -> None:
    """Retrieves and saves conversations that contain
    the tweets in a sampled file. The files are saved
    in the sampled_conversations folder. Uses the full
    archive search endpoint.

    Retweets are not considered conversations; the
    retweeted tweets are instead taken as the root.
        
    Args:
        - sample_file: path to the sampled tweets. The file should be
        the output of the sampled stream in .jsonl format, with the
        first attribute 'data' for each line (default when sampling
        from the command line).
        
    No return value.
    """
    
    # TODO: use 'yield' to generate files only once
    sample = read_file(sample_file)
    print('Retrieving conversations for {} tweets.\
    With 1200 queries per hour this should take {:.3f} minutes ({:.3f} hours)'.format(len(sample), len(sample)/30, len(sample)/1800))
    t1 = pytime.time()
    tot_conv_retrieved = 0
    
    for i,t in enumerate(sample):
        
        conv_id = t['data']['conversation_id']
        
        # If the sampled tweet is a RT, use the retweeted tweet as root
        if 'referenced_tweets' in t['data']:
            for ref in t['data']['referenced_tweets']:
                if ref['type'] == 'retweeted':
                    conv_id = ref['id']
                    break
        else:
            logging.info(f'Tweet {conv_id} has no referenced tweets.')
                        
        if os.path.isfile(f'sampled_conversations/{conv_id}_conversation-tweets.jsonl'):
            logging.info(f'Conversation {conv_id} already retrieved.')
            continue
            
        # Fetch root; if retrieval fails, skip the conversation.
        root_exists = get_single_root(conv_id)
        if not root_exists:
            logging.info(f'Root tweet {conv_id} could not be retrieved.')
            continue
        
        # Calling full archive search (rate limit: 300 requests/15 minutes)
        conv_query = f'conversation_id:{conv_id}'
        conv_tweets = full_archive_search(query_=conv_query, max_results_pp=100,
                                          expansions_='author_id,in_reply_to_user_id,referenced_tweets.id,referenced_tweets.id.author_id,entities.mentions.username',
                                          tweet_fields_='attachments,author_id,context_annotations,conversation_id,created_at,entities,id,public_metrics,text,referenced_tweets,reply_settings',
                                          user_fields_='created_at,entities,id,name,protected,public_metrics,url,username',
                                          media_fields_='type', poll_fields_='id', place_fields_='id')

        filename = f'sampled_conversations/{conv_id}_conversation-tweets.jsonl'
        n_results = len(conv_tweets)
        with open(filename, 'a+') as filehandle:
            for tweet in conv_tweets:
                filehandle.write('%s\n' % json.dumps(tweet))
        logging.info(f'conversation {conv_id} resulted in a total of {n_results} results.')
        if n_results > 0:
            tot_conv_retrieved += 1
    
    t2 = pytime.time() - t1
    logging.info(f'Conversation retrieval took {t2} seconds. Out of {len(sample)} conversations, {tot_conv_retrieved} contained replies.')
    print('Finished retrieving {} conversations in {:.3f} minutes ({:.3f} hours)'.format(tot_conv_retrieved, t2/60, t2/3600))
    
    return


def get_root_tweets(conv_ids: list[str], get_saved=False) -> list[dict]:
    """Returns a list of tuples with information on a
    list of tweets (tweetIDs). Use case is to retrieve
    the id of the root of a conversation in order to query
    for retweets.
    
    Args:
        - conv_ids: list of tweet or conversation IDs
        - get_saved: boolean; function will return the
        root tweets saved in 'root_tweets/' if True.
        
    Returns:
        - roots: list of tuples containing the author ID, engagement
        metrics, tweet text and time stamp of the original tweet.
        - retrieved_conv_ids: list of retrieved conversation IDs;
        some might be missing due to tweets being inaccessible
    """   
    roots = []
    retrieved_conv_ids = []
    
    if get_saved:
        for conv_id in conv_ids:
            if os.path.isfile(f'root_tweets/{conv_id}_root.jsonl'):
                retrieved_conv_ids.append(conv_id)
                orig_tweet = read_file(f'root_tweets/{conv_id}_root.jsonl')[0]
                roots.append({'id':         orig_tweet['author_id'],
                              'n_retweets': orig_tweet['public_metrics']['retweet_count'],
                              'text':       orig_tweet['text'],
                              'created_at': orig_tweet['created_at']})
    else:
        # Fetch missing only
        n_res = 0
        results = client.tweet_lookup(filter(root_tweet_missing, conv_ids))
        for page in results:
            res = expansions.flatten(page)
            for orig_tweet in res:
                cid = orig_tweet['conversation_id']
                append_objs_to_file(f'root_tweets/{cid}_root.jsonl', [orig_tweet])
                n_res += 1
        
        print(f'Retrieved {n_res} new roots.')
        
        # Retrieve all saved roots
        roots, retrieved_conv_ids = get_root_tweets(conv_ids, True)
        
    return roots, retrieved_conv_ids


def root_tweet_missing(conv_id):
    """Returns True if root tweet with ID == conv_id is missing."""
    return not os.path.isfile(f'root_tweets/{conv_id}_root.jsonl')


def get_saved_conversation_ids(lower_bound: int, upper_bound: int, folder='sampled_conversations') -> list:
    """Returns a list of conversation ids between a given
    lower and upper bound from the folder of saved conversations.
    
    Args:
        - lower_bound: integer lower limit of conversation IDs to include
        - upper_bound: integer upper limit of conversation IDs to include
        - folder: folder to search for conversations in
    
    Returns:
        - conv_ids: a list of conversation IDs from the folder
    """
    
    file_paths = os.listdir(folder)
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
        if os.path.isfile(f'retweets/{conv_ids[i]}.jsonl'):
            continue
        
        retweets = []
        
        # Find retweeting authors
        retweeters = get_retweeters(conv_ids[i])
        for retweeter in retweeters:
            # Query for any retweet by author of original tweet author, containing the same text
            # Should only return 1 result in practice (if querying on text)
            root_author = root['id']
            text = root['text']
            #retweets_result = recent_search(query_=f'retweets_of:{root['id']} from:{retweeter} "{text}"')
            retweets_result = recent_search(query_=f'retweets_of:{root_author} from:{retweeter}')
            
            for rt in retweets_result:
                # Find relevant results: filter on referenced_tweets
                # Should only contain reference to 1 tweet if it is a pure retweet;
                # if so use rt['referenced_tweets'][0]
                for t in rt['referenced_tweets']:
                    if t['id'] == conv_ids[i]:
                        # TODO: consider adding the RT to a file in this loop;
                        # when dealing with thousands of RTs we cannot abort the
                        # function call without losing all the data!
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


def not_space(s: str) -> bool:
    """Returns false if s is empty or
    consists of whitespace characters.
    
    Args:
        - s: string to check
    
    Returns:
        - boolean indicating if string
        fulfills the criterion
    """
    return not (s.isspace() or not s)

def handle_special(s: str) -> str:
    """Removes quotation marks and special
    characters (such as ampersands etc.) from
    the string where they would cause an error
    if passed to the API. Used when fetching
    retweets and querying on the tweet text.
    Seems to work for right-to-left languages
    as well, e.g., Arabic and Hebrew.
    
    Args:
        - s: string that contains quotation
        marks (" or '), and a hyperlink removed.
        
    Returns:
        - a string with quotation marks in the
        correct places: only between characters
        in the string, and no quotation marks at
        the end or beginning of the string.
    """
    #s_list = s.split('"')
    s_list = re.split('\'|"|&amp;|&gt;|&lt;', s)
    # TODO/Decision: Heuristic: If there is a string that has more than 20 characters in it
    # it alone may be used as a query to minimize the risk of other special characters
    # messing up the query
    return '" "'.join(filter(not_space, s_list))


def get_all_retweets(conv_ids: list[str]) -> None:
    """Get retweets of conversation roots and stores them in a .jsonl-file.
    Will ignore the conversation IDs that already is associated with a 
    retweet file. Efficient query for all RTs containing the text, and is
    an RT of the root author. In practice, not all retweets can be fetched
    due to some profiles being protected.
        
    Args:
        - conv_ids: list of conversation IDs to fetch retweets for
        
    No return value.    
    """

    roots, retrieved_conv_ids = get_root_tweets(conv_ids, get_saved=True)
    
    n_roots = len(roots)
    n_retweets = 0
    roots_skipped, no_text_roots = 0, 0
    for r in roots:
        n_retweets += r['n_retweets']
    
    
    print('{} of {} root tweets were available'.format(n_roots, len(conv_ids)))
    print('Total of {} retweets from {} roots.\
    With 300 queries per hour this should take {:.3f} minutes ({:.3f} hours)'.format(n_retweets, n_roots, n_roots/5, n_roots/300))
    t1 = pytime.time()
    
    for i, root in enumerate(roots):
        if os.path.isfile(f'retweets/{retrieved_conv_ids[i]}.jsonl'):
            roots_skipped += 1
            continue
        
        # DECISION: Heuristic to get rid of links in tweets
        root_text = root['text'].split('https://t.co/')[0]
        # DECISION: We could also take root_text = ''.join(root['text'].split('https://t.co/')[:-1])
        # to remove the last link (alt. use rfind())
        
        # Handle quotation marks
        if root_text.find('"') != -1:
            root_text = handle_special(root_text)
            print(f'Handled quotation marks in conversation {retrieved_conv_ids[i]}; text:', root_text)
        elif not root_text:
            write_text('skipped_retweets.txt', 'skipping conversation {} (text: {})'.format(retrieved_conv_ids[i], root['text']))
            no_text_roots += 1
            continue
        
        root_author = root['id']
        
        # TODO: Modify the expansion fields as well? See twitter or Postman WS for allowed values
        # TODO: Use try catch to avoid breaking HTTPerrors?
        #results = recent_search(query_=f'retweets_of:{root_author} "{root_text}"',
        #                        max_results_pp=100, tweet_fields_='id,author_id,referenced_tweets,created_at',
        #                        user_fields_='id')
        results = full_archive_search(query_=f'retweets_of:{root_author} "{root_text}"', max_results_pp=500,
                                      tweet_fields_='id,author_id,referenced_tweets,created_at', user_fields_='id',
                                      media_fields_='type', poll_fields_='id', place_fields_='id')
        
        # Debug prints
        #NRT = root['n_retweets']
        #print(f'full archive search yielded {len(results)} results; the root has {NRT} retweets')
        
        retweets = []
        for retweet in results:
            for t in retweet['referenced_tweets']:
                if t['id'] == retrieved_conv_ids[i]:
                    # Append to file here? How many results do we expect?
                    retweets.append(retweet)
                    break
                    
        # TODO: warn if there are many retweets according to root, but none were retrieved in the query
        # Add the expected number of RTs (root['n_retweets']) to the log file
        num_rts = root['n_retweets']
        append_objs_to_file(f'retweets/{retrieved_conv_ids[i]}.jsonl', retweets)
        logging.info(f'Retrieved {len(retweets)} (of {num_rts}) retweets to conversation {retrieved_conv_ids[i]}.')
        
    t2 = pytime.time() - t1
    logging.info(f'Retweets retrieval took {t2} seconds.')
    print('Retweets retrieval took {:.3f} minutes ({:.3f} hours); '.format(t2/60, t2/3600))
    print('Skipped {} roots for which RTs are already retrieved, and {} roots which contained no searchable text.'.format(roots_skipped, no_text_roots))
    
    return
    

def remove_missing_rt_files() -> None:
    """Checks for retweet files that should contain something,
    but are empty. Puts the missing files in another folder.
    
    No args or return value. Results are logged.
    """
    
    for f in os.listdir('retweets'):
        c_id = f.split('.')[0]
        size = os.stat(f'retweets/{f}').st_size
        root = read_file(f'root_tweets/{c_id}_root.jsonl')[0]
        n_rts = root['public_metrics']['retweet_count']

        if size == 0 and n_rts != 0:
            os.rename(f'retweets/{f}', f'missing_rts/{f}')
            logging.info(f'moved retweets/{f} to missing_rts/{f}')
        else:
            logging.info(f'retweets/{f} of size {size} (with {n_rts} retweets) not moved')


def compute_relative_time(t0: str, t: str) -> float:
    """Returns the time offset in hours of a time point t
    relative to a reference t0.
    
    Args:
        - t0: reference time point (in format %Y-%m-%dT%H:%M:%S.000Z)
        - t: time point of interest (in format %Y-%m-%dT%H:%M:%S.000Z)
    
    Returns:
        - dt: time offset in hours
    """
    t0 = datetime.datetime.strptime(t0, '%Y-%m-%dT%H:%M:%S.000Z')
    t = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.000Z')
    
    # timedelta objects have attributes days, seconds and microseconds
    t_delta = t-t0
    dt = (86400*t_delta.days + t_delta.seconds)/3600
    return dt


def auto_correlate(x, symmetric=True):
    """Returns the autocorrelation of x: ac[x,k] = sum_n x[n+k]x[n]
    
    Args:
        - x: series to correlate
        - symmetric: boolean indicating whether or not to return
        the whole (symmetric) result, or just the right half.
        
    Returns:
        - ac: The result of the auto correlation
    
    """
    ac = np.correlate(x, x, 'full')
    if not symmetric:
        ac = ac[len(x)-1:]
        ac = ac/ac[0]
    else:
        ac = ac/ac[len(x)-1]
    return ac


def mean_square_error_fo(bin_values, beta: float, lambda_: float) -> float:
    """Computes the mean square error of a first order
    linear model with parameters beta and lambda_, with
    data discretized in steps of 1 unit of time.
    
    Args:
        - bin_values: bin values of the engagement histogram
        - beta: parameter modelling the initial response to
        the tweet. Should be positive
        - lambda_: decay constant which should take values
        less than 1.
        
    Returns:
        - MSE: the mean square error
    """
    x_hat = beta
    MSE = 0
    for x in bin_values:
        MSE += (x-x_hat)**2
        x_hat *= (1-lambda_)
    MSE = MSE / len(bin_values)
    return MSE


def eval_error(time, model_engagement, true_engagement):
    """Computes the mean square error, and the ratio between the sum of
    squared residuals and the sum of the squared signal in continuous time
    
    Args:
        - time: time vector for the observations
        - true_engagement: engagement at times
        specified in time vector
        - lambda_: (optimal) decay constant
        - beta_: parameter modelling the initial
        response to the tweet
        
    Returns:
        - MSE: mean square error
        - RSS_frac
        - model_engagement: 
    """
    res = model_engagement - true_engagement
    sq_residual = np.square(res)
    sq_signal = np.square(true_engagement)
    
    MSE = np.mean(sq_residual)
    RSS_frac = np.sum(sq_residual)/np.sum(sq_signal)
    
    #plt.hist(res, bins=200)
    #plt.show()
    
    return MSE, RSS_frac


def exponential(x_, lambda_, beta_):
    """Exponential decay with parameters lambda and beta."""
    return beta_*np.exp(-lambda_*x_)


def power(x_, lambda_, beta_):
    """Power law."""
    return beta_*(np.power(np.array(x_, dtype=float), -lambda_))


def estimate_decay_parameters(time, engagement):
    """Returns an estimate of the parameters of the
    exponential decay function of engagement over time.
    
    Args:
        - time: time vector
        - engagement: engagement at times specified
        in time vector
        
    Returns:
        - popt: optimal parameters
    """
    # lower and upper bounds on lambda and beta:
    # 1e-4 <= lambda <= 1e3, and 1 <= beta <= 1e9.
    bounds_ = ([1e-4, 1], [1e3, 1e9])
    init = [1, 1]
    
    # Optimization: Trust Region Reflective algorithm 'trf'
    # Levenberg-Marquardt ('lm') does not handle parameter bounds
    method_ = 'trf'
    
    popt, _ = curve_fit(exponential, time, engagement, p0=init,
                     bounds=bounds_, method=method_)
    return popt


def delayed_biexponential(x_, alpha, beta, gamma, eta, sigma, delay):
    """Solution to second order system. Assumes that the time bins
    are spaced 1 hour apart (tau_d is the delay given in hours).
    
    Note that gamma and alpha are positive, whereas they are negative
    in the original equation system.
    """
    #tau_dn = int(tau_d*100)
    #u_delay = np.concatenate((np.zeros(tau_dn), np.ones(len(x_)-tau_dn)))
    #return eta*np.exp(-alpha*x_) + (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-tau_d))*u_delay - (beta/(gamma-alpha))*np.exp(-gamma*(x_-tau_d))*u_delay 
    delay=int(delay)
    delayed = (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-delay)) - (beta/(gamma-alpha))*np.exp(-gamma*(x_-delay))
    delayed[:delay] = 0
    return eta*np.exp(-alpha*x_) + delayed


def biexponential(x_, alpha, beta, gamma, rho):
    """Solution to system
    dx1/dt = alpha*x1(t) + beta*x2(t)
    dx2/dt = gamma*x2(t) + rho*d(t)
    """
    return np.exp(-alpha*(x_))*rho*beta/(gamma-alpha) + np.exp(-gamma*(x_))*rho*(1-(beta/(gamma-alpha)))


def estimate_biexponential(time, engagement, loss_='linear'):
    
    def biexponential_opt(x_, alpha, beta, d_gamma, rho):
        """Solution to system
        dx1/dt = alpha*x1(t) + beta*x2(t)
        dx2/dt = gamma*x2(t) + rho*d(t)
        """
        gamma = alpha + d_gamma
        return np.exp(-alpha*(x_))*rho*beta/(gamma-alpha) + np.exp(-gamma*(x_))*rho*(1-(beta/(gamma-alpha)))

    method_ = 'trf'
    bounds_ = ([1e-5, 1e-2, 1e-5, 1e-3],
               [1e3,  1e9,  1e3,  1e9])
    init = [1e-1, 1, 1e-1, 1]
    popt, _ = curve_fit(biexponential_opt, time, engagement,
                        p0=init, bounds=bounds_, method=method_, loss=loss_)
    a, b, d_g, r = popt[0], popt[1], popt[2], popt[3]
    g = a + d_g
    return a, b, g, r


def estimate_second_order(time, engagement, delay, loss_='linear'):
    """Fit a second order model of type
    dx1/dt = ax1 + bx2 + nd(t) + sd(t-td)
    dx2/dt = gx2 +     + rd(t-td)
    Since r and b are coupled we set r=1
    
    Args:
        - time: time vector
        - engagement: engagement at times specified
        in time vector
        
    Returns:
        - a, b, g, n, s: model parameters
    """
    
    def biexponential_free_delay(x_, d_alpha, beta, gamma, eta, sigma, delay):
        """Solution to second order system. Assumes that the time bins
        are spaced 1 hour apart (tau_d is the delay given in hours).

        Note that gamma and alpha are positive, whereas they are negative
        in the original equation system.
        """
        #tau_dn = int(tau_d*100)
        #u_delay = np.concatenate((np.zeros(tau_dn), np.ones(len(x_)-tau_dn)))
        #return eta*np.exp(-alpha*x_) + (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-tau_d))*u_delay - (beta/(gamma-alpha))*np.exp(-gamma*(x_-tau_d))*u_delay 
        delay_n=int(delay)
        alpha = gamma + d_alpha
        delayed = (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-delay)) - (beta/(gamma-alpha))*np.exp(-gamma*(x_-delay))
        delayed[:delay_n] = 0
        return eta*np.exp(-alpha*x_) + delayed
    
    def biexponential_fixed_delay(x_, d_alpha, beta, gamma, eta, sigma):
        """Solution to second order system. Assumes that the time bins
        are spaced 1 hour apart (delay is given by encapsulating function hours).

        Note that gamma and alpha are positive, whereas they are negative
        in the original equation system.
        """
        #tau_dn = int(tau_d*100)
        #u_delay = np.concatenate((np.zeros(tau_dn), np.ones(len(x_)-tau_dn)))
        alpha = gamma + d_alpha
        delayed = (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-delay)) - (beta/(gamma-alpha))*np.exp(-gamma*(x_-delay))
        delayed[:delay] = 0
        return eta*np.exp(-alpha*x_) + delayed
        # return eta*np.exp(-alpha*x_) + (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-tau_d))*u_delay - (beta/(gamma-alpha))*np.exp(-gamma*(x_-tau_d))*u_delay 

    method_ = 'trf'
    if delay != 0:
        bounds_ = ([1e-5, 1e-2, 1e-5, 1,   1e-5],
                   [1e3, 1e9,   1e3,  1e9, 1e6])
        init = [1e-1, 1, 1e-1, 2, 1]
        popt, _ = curve_fit(biexponential_fixed_delay, time, engagement,
                            p0=init, bounds=bounds_, method=method_, loss=loss_)
        d_a, b, g, n, s, td = popt[0], popt[1], popt[2], popt[3], popt[4], 0
        
    else:
        tau_max = int(time[-1]-time[0])
        # instead of fitting gamma, use auxillary parameter c=alpha-gamma, which is
        # always less than 0 (if alpha and gamma are defined negative) and describes the
        # discrepancy between the types of decay. Using gamma < alpha < 0 will give an error.
        bounds_ = ([1e-5, 1e-2, 1e-5, 1, 1e-5, 0],
                   [1e3, 1e9,   1e3,  1e9, 1e6, tau_max])
        init = [1e-1, 1, 1e-1, 2, 1, int(0.5*tau_max)]

        # Linear fit (alt. L1 fit with 'soft_l1')
        popt, _ = curve_fit(biexponential_free_delay, time, engagement,
                            p0=init, bounds=bounds_, method=method_, loss=loss_)
        d_a, b, g, n, s, td = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    
    a = g + d_a
    return a, b, g, n, s, td


def peak_detection(e, lag, threshold, influence):
    """Peak detection algorithm using deviation from moving median to
    find peaks in histograms.
    
    From: J.P.G. van Brakel. Robust peak detection algorithm (using z-scores), 2019.
    https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtimetimeseries-data.
    
    Args:
        - e: engagement histogram values
        - lags: size of the moving window
        - threshold: number of standard deviations required to trigger detection
        - influence: influence of the next sample to the mean
    
    Returns:
        - A dictionary that contains the peak detection signal: with elements
        in {1-,0,1}, indicating a negativ peak, no peak, or positive peak; the
        moving median, and the moving standard deviation.
    """
    
    signals = np.zeros(len(e))
    filteredE = np.array(e)
    avgFilter = [0]*len(e)
    stdFilter = [0]*len(e)
    avgFilter[lag - 1] = np.mean(e[0:lag])
    stdFilter[lag - 1] = np.std(e[0:lag])
    
    # For each data point
    for i in range(lag, len(e) - 1):
        
        # Detect a deviation from the median +/- threshold*std
        if abs(e[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            if e[i] > avgFilter[i-1]:
                signals[i] = 1 # positive peak
            else:
                signals[i] = -1 # negative peak

            # Less influence from peak data
            filteredE[i] = influence * e[i] + (1 - influence) * filteredE[i-1]
        else:
            # No peak detected, no less influence
            signals[i] = 0
            filteredE[i] = e[i]
        
        avgFilter[i] = np.mean(filteredE[(i-lag):i])
        stdFilter[i] = np.std(filteredE[(i-lag):i])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter),
                lag=lag,
                threshold=threshold,
                influence=influence)


def filter_peaks(result, y, peak_threshold: float, adj_dist=1):
    """Filters out the desired peaks from the result of the peak
    detection algorithm. Returns the peak coordinates.
    
    Args:
        - result: the result dictionary of the peak detection algorithm
        - y: engagement histogram values
        - peak_threshold: fraction of global maximum a peak must attain
        in order not to be filtered out
        - adj_dist: max distance between adjacent peaks; these will be
        reduced to the highest peak
        
    Returns:
        - peaks_x: x-coordinate for the relevant peaks
        - peaks_y: y-coordinate for the relevant peaks
    """
    
    xx, yy= [], []
    
    assert peak_threshold >= 0 and peak_threshold <= 1
    glob_max = max(y)
    
    # Filter out peaks below the global threshold
    for i,r in enumerate(result["signals"]):
        if r==1 and y[i] > peak_threshold*glob_max:
            xx.append(i+1)
            yy.append(y[i])

    #print(xx, yy)
    peaks_x, peaks_y = [], []
    
    prev_x = xx[0]
    segment_x, segment_y = [xx[0]], [yy[0]]
    curr_max, curr_max_y = xx[0], yy[0]

    # Filter out adjacent peaks
    for i in range(1, len(xx)):
        if xx[i] > prev_x + adj_dist:
            #print(f'{xx[i]}!={prev_x+adj_dist}; peak = ({curr_max},{curr_max_y})')
            peaks_x.append(curr_max)
            peaks_y.append(curr_max_y)
            
            segment_y = [yy[i]]
            segment_x = [xx[i]]
            curr_max = xx[i]
            curr_max_y = yy[i]
        else:
            if curr_max_y < yy[i]:
                curr_max = xx[i]
                curr_max_y = yy[i]
            segment_x.append(xx[i])
            segment_y.append(yy[i])
            
        prev_x = xx[i]
            
    peaks_x.append(curr_max)
    peaks_y.append(curr_max_y)
    
    valleys = []
    for i in range(1,len(peaks_x)):
        if peaks_x[i] > peaks_x[i-1] + 3:
            valleys.append((peaks_x[i-1],peaks_x[i]))
    #print(valleys)
    #print(peaks_x, peaks_y)
    
    return peaks_x, peaks_y # xx, yy


def plot_peak_detection(y, result, conv_id):
    """Plots the peaks from the raw result of the detection algorithm.
    The result data is filtered before plotting.
    
    Args:
        - y: the engagement histogram values
        - result: result dictionary of the peak detection algorithm
        - conv_id: conversation id
        
    Returns:
        - type_: the type of engagement graph, first position
        indicates delay (1X) or no delay (0X), the second position
        indicates a single peak (0), double peak (2), bump (3), or
        multiple peaks (4).
        - first_peak: bin number of the first peak (i.e., delay).
        Chosen as the largest of the first two peaks, or -1 in
        the odd case that there is no peak.
    """
    
    peaks_x, peaks_y = filter_peaks(result, y, peak_threshold=0.15, adj_dist=2)
    
    # DECISION: We use the heuristic to pick the largest of the two first peaks
    type_ = 0
    if len(peaks_x) != 0:
        if peaks_x[0] <= result['lag']+1:
            type_ += 0 # NOP
        else:
            type_ += 10
        
        if len(peaks_x) == 1:
            first_peak = peaks_x[0] - result['lag'] - 1
            second_peak = -1
        else:
            if peaks_y[0] > peaks_y[1]:
                first_peak = peaks_x[0] - result['lag'] - 1
                second_peak = peaks_x[1] - result['lag'] - 1
            else:
                first_peak = peaks_x[1] - result['lag'] - 1
                if len(peaks_x) > 2:
                    second_peak = peaks_x[2] - result['lag'] - 1
                else:
                    second_peak = -1
        
        if (len(peaks_x) == 2) and (min(peaks_y[0], peaks_y[1]) > 0.45*max(peaks_y[0], peaks_y[1])):
            type_ += 1
        elif len(peaks_x) == 2:
            type_ += 2
        elif len(peaks_x) > 2:
            type_ += 3
    else:
        first_peak = -1
        second_peak = -1
        type_ = 4
    
    
    plt.figure(figsize=(8,8), clear=True)
    #plt.subplot(211)
    plt.plot(np.arange(1, len(y)+1), y, color='navy', lw=2)

    plt.plot(np.arange(1, len(y)+1),
               result["avgFilter"], '--', color='gold', lw=1)

    plt.plot(peaks_x, peaks_y, 'o', color='red')

    plt.plot(np.arange(1, len(y)+1),
               result["avgFilter"] + result['threshold'] * result["stdFilter"], color="sienna", lw=0.8)

    plt.plot(np.arange(1, len(y)+1),
               result["avgFilter"] - result['threshold'] * result["stdFilter"], color="sienna", lw=0.8)

    #plt.subplot(212)
    suppressed = [i for i,r in enumerate(result['signals']) if r > 0]
    supp_y = -1*np.ones(len(suppressed))
    plt.plot(suppressed, supp_y, 'o', color='salmon')
    #plt.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
    #plt.ylim(-1.5, 1.5)
    plt.savefig(f'sampled_conversations_graphs/peak_detection/{type_}/{conv_id}_peaks2.png')
    plt.close('all')

    return type_, first_peak, second_peak


def get_file_paths(conv_id: str):
    """Returns the file paths to the conversation root, replies, and retweets
    for a given conversation ID.
    
    Args:
        - conv_id: conversation ID
        
    Returns:
        - root_path, conv_path, retw_path: file paths
    """
    
    root_path = f'root_tweets/{conv_id}_root.jsonl'
    conv_path = f'sampled_conversations/{conv_id}_conversation-tweets.jsonl'
    retw_path = f'retweets/{conv_id}.jsonl'
    
    return root_path, conv_path, retw_path


def process_engagement_times(conv_ids: list[str], delta_sec: float) -> None:
    """Retrieves the engagement times of retweets and replies.
    Only considers conversations where root, conversation, and
    retweet files exist.
    
    Args:
        - conv_ids: list of conversation IDs
        - delta_sec: the time step length in seconds
        of the time series discretization in seconds
        
    No return value.
    """

    out_conv_id = []
    sufficient, missing_data, too_few, opt_failed = 0, 0, 0, 0
    
    # Iterate with generator from filter(all_files_exist, conv_ids)?
    for conv_id in conv_ids:
        root_path, conv_path, retw_path = get_file_paths(conv_id)
        
        # Check that root, conversation and retweet files exist.
        if os.path.isfile(root_path) and os.path.isfile(conv_path) and os.path.isfile(retw_path):
            root = read_file(root_path)[0]
            out_conv_id.append(conv_id)
        else:
            missing_data += 1
            continue
        
        """If plotting the number of followers of those who interact by retweeting
        skip = False
        for rt in file_generator(retw_path):
            if not 'public_metrics' in rt['author']:
                skip = True
            break
        if skip:
            missing_data += 1
            continue
        """
        
        engagement_time, followers = [], []
        rep_time, rt_time = [], []
        replying_users = set()
        
        for re in file_generator(conv_path):
            re_t = compute_relative_time(root['created_at'], re['created_at'])
            engagement_time.append(re_t)
            rep_time.append(re_t)
            replying_users.add(re['author_id'])
            #followers.append(re['author']['public_metrics']['followers_count'])
        
        n_replies = len(engagement_time)
        
        for rt in file_generator(retw_path): 
            rt_t = compute_relative_time(root['created_at'], rt['created_at'])
            engagement_time.append(rt_t)
            rt_time.append(rt_t)
            #followers.append(-1)
            #followers.append(rt['author']['public_metrics']['followers_count'])
        
        # TODO/DECISION: Ignore the conversations that have fewer than 50 replies/retweets,
        # or perhaps only engagament in the first X minutes
        tot_eng = len(engagement_time)
        n_api_retweets = root['public_metrics']['retweet_count']
        
        # if tot_eng < 50:
        if (tot_eng < 50) or ((tot_eng-n_replies) <= 1e-5*n_api_retweets):
            too_few += 1
            continue
        else:
            sufficient += 1
        
        
        reply_ratio = n_replies / tot_eng
        
        # Bin and estimate engagement curve (time series with increments of delta_t)
        mx = np.max(engagement_time)    # max time in hours
        delta_h = delta_sec/3600        # delta_t in hours
        n_bins = int((mx//delta_h) + 2) # Add 2 to compensate for integer division and endpoint in linspace
        bins = np.linspace(start=0, stop=n_bins*delta_h, num=n_bins, endpoint=False, retstep=False, dtype=None, axis=0)
        
        n, bs = np.histogram(engagement_time, bins)
        
        lag = 10
        threshold = 1.5
        influence = 0.8
        # rng = np.random.default_rng(123)
        rd = np.zeros(lag) # np.abs(rng.standard_normal(lag))
        time_series = np.concatenate((rd,n))#/max(n)
        result = peak_detection(time_series, lag=lag, threshold=threshold, influence=influence)
        type_, first_peak, second_peak = plot_peak_detection(time_series, result, conv_id, plot=False)
        
        root_followers = root['author']['public_metrics']['followers_count']
        n_api_replies = root['public_metrics']['reply_count']
        
        #acE = auto_correlate(n, symmetric=False)
        #create_plot(x=acE, y=None, path=f'sampled_conversations_graphs/engagement_correlation/{conv_id}_corr.png',
        #            format_='-', title='Autocorrelation of engagement', xlab=f'time (dt={delta_sec} seconds)', ylab='correlation')
        
        t = (bins[1:] + bins[:-1])/2 # Time vector (center of bins, can also use bins[:-1])
        
        try:
            # Take the delay into account (TODO: also in the MSE computation)
            #if type_ >= 10:
            if first_peak > 0:
                t = t[first_peak:] - delta_h*first_peak # = t[:-first_peak]?
                n = n[first_peak:]
            else:
                first_peak = 0
                

            """Fit only bins until 97% of engagement is reached, useful when using log loss
            cum_eng = np.cumsum(n)
            stop_ix = -1
            for i,e in enumerate(cum_eng):
                if e > cum_eng[-1]*0.97:
                    stop_ix = i
                    break
            
            n = n[:stop_ix]
            t = t[:stop_ix]
            """
            
            a, b, g, eta, s = 'None', 'None', 'None', 'None', 'None'
            lambda_, beta_ = 'None', 'None'
            a, b, g, rho = estimate_biexponential(time=t, engagement=n, loss_='linear')
            model_eng = biexponential(t, alpha=a, beta=b, gamma=g, rho=rho)
            
            """
            if second_peak == -1:
                # Set delay to zero in second order system to get a biexponential solution
                #opt_params = estimate_decay_parameters(t, n, loss_='linear', f_scale_=1.0)
                #lambda_, beta_ = opt_params[0], opt_params[1]
                #lambda_, beta_ = estimate_lin_decay(t, n) # log loss
                
                a, b, g, eta, s, delay = estimate_second_order(time=t, engagement=n, delay=0, loss_='linear')
                model_eng = delayed_biexponential(t, alpha=a, beta=b, gamma=g, eta=eta, sigma=s, delay=delay)
                #model_eng = exponential(t, lambda_, beta_)
                #a, b, g, eta, s = 'None', 'None', 'None', 'None', 'None'
            else:
                delay = second_peak - first_peak
                a, b, g, eta, s, delay = estimate_second_order(time=t, engagement=n, delay=delay, loss_='linear')
                delay=int(delay)
                model_eng = delayed_biexponential(t, alpha=a, beta=b, gamma=g, eta=eta, sigma=s, delay=delay)
                lambda_, beta_ = 'None', 'None'
            """
            MSE, RSS_frac = eval_error(t, model_eng, n)
            
            result_file = 'parameter_estimations/estimations_delay_biexp-no-impulse-delay_L2.txt'
            write_text(file_name=result_file,
                       text=f'{conv_id},{root_followers},{len(replying_users)},{type_},{first_peak},{second_peak},{lambda_},{beta_},{a},{b},{g},{rho},{MSE},{RSS_frac},{n_replies},{reply_ratio},{tot_eng},{n_api_retweets},{n_api_replies}')
                       #text=f'{conv_id},{root_followers},{len(replying_users)},{type_},{first_peak},{second_peak},{lambda_},{beta_},{a},{b},{g},{eta},{s},{MSE},{RSS_frac},{n_replies},{reply_ratio},{tot_eng},{n_api_retweets},{n_api_replies}')
                       #text=f'{conv_id},{root_followers},{len(replying_users)},{type_},{first_peak},{lambda_},{beta_},{MSE},{RSS_frac},{n_replies},{reply_ratio},{tot_eng},{n_api_retweets},{n_api_replies}')
            
            graph_path_line = f'sampled_conversations_graphs/engagement_histograms/{conv_id}_repl_retw_ds{int(delta_sec)}_delay_stacked_biexp_L2.png'
            _, _ = create_hist([rep_time, rt_time], bins, path=graph_path_line,
                               #engagement_time, bins, path=graph_path_line,
                               title='Engagement over time (replies and RTs)',
                               xlab='time (h)', ylab='counts (replies and retweets)',
                               log_=False, overlay_line=True, t=t+delta_h*first_peak, y=model_eng)
            
            #_, _ = create_hist_scatter(engagement_time, bins, path=f'sampled_conversations_graphs/engagement_histograms/interactor_followers/{conv_id}_engag_ds{int(delta_sec)}_type3_flws.png',
            #                           title='Engagement over time (replies and RTs)', xlab='time (h)', ylab='counts (replies and retweets)', log_=False,
            #                           overlay_line=True, t=t+delta_h*first_peak, y=model_eng, scatter_y=followers, root_flw=root['author']['public_metrics']['followers_count'])
            
        except Exception as e:
            logging.warning(e)
            logging.info(f'optimization failed for conversation {conv_id}: replies:{n_replies}, reply_ratio:{reply_ratio}')
            sufficient -= 1
            opt_failed += 1
        
    print('Plots: {}, missing data: {}, too few data points: {}, optimization failed: {}'.format(sufficient, missing_data, too_few, opt_failed))
    
    return


def undirected_message_tree(conv_id: str) -> nx.Graph:
    """Returns an undirected graph (tree) that represents the
    messages in a conversation. The graph is empty if any of
    the root or conversation files are missing. The graph is
    not guaranteed to be connected.
    
    Args:
        - conv_id: a single conversation ID
        
    Returns:
        - G: a conversation graph
    """
    
    root_path = f'root_tweets/{conv_id}_root.jsonl'
    conv_path = f'sampled_conversations/{conv_id}_conversation-tweets.jsonl'
    
    G = nx.Graph()
    
    if not os.path.isfile(root_path) or not os.path.isfile(conv_path):
        return G
    
    for re in file_generator(conv_path):
        for ref_tw in re['referenced_tweets']:
            if ref_tw['type'] == 'replied_to':
                G.add_edge(re['id'], ref_tw['id'])
                break
    return G


def compute_radii(conv_ids: list[str]) -> list[int]:
    """Computes and saves the radii of conversations graphs,
    i.e., the maximum depth of the conversation tree. Ignores
    the conversations that fewer than 90% of nodes connected
    to the root. The result is stored in graph_radius.txt in
    the parameter_estimations folder.
    
    Args:
        - conv_ids: list of conversation IDs
        
    Returns:
        - rad: vector of conversation radii
    """
    rad = []
    connected = 0
    for c_id in conv_ids:
                        
        G = undirected_message_tree(c_id)
        
        if not G.has_node(c_id):
            continue
        
        n_nodes = len(list(G.nodes))
        # Allow for unconnected graphs if the component that contains
        # the root tweet has more than 90% of all nodes.
        if n_nodes > 0:
            if nx.is_connected(G):
                r = nx.radius(G)
                rad.append(r)
                write_text('parameter_estimations/graph_radius.txt', f'{c_id},{nx.radius(G)}')
                connected += 1
            else:
                nodes_largest_cc = nx.node_connected_component(G, c_id)
                if len(nodes_largest_cc) >= 0.9*n_nodes:
                    r = nx.radius(G.subgraph(nodes_largest_cc))
                    rad.append(r)
                    write_text('parameter_estimations/graph_radius.txt', f'{c_id},{r}')
                    
    print("{} conversations graphs were not sufficiently connected. Obtained results from {} conversations, of which {} were completely connected.".format(len(conv_ids)-len(rad), len(rad), connected))
    return rad


def retweet_metrics(conv_ids: list[str]) -> None:
    """Computes retweet metrics and print to a file.
    Plots retweets over time from the initial post.
    Saves images to 'sampled_conversation_graphs' folder.
    
    NOTE: Implicitly assumes that get_root_tweets() has
    been called through get_retweets_of() for these
    conversation IDs.
    
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
        # Root is poster first, no further adjustments to time needed
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
            # DECISION: return if no retweets?
            # return
        
        
        create_hist(engagement_time, bins=50, path=f'sampled_conversations_graphs/{conv_id}_retweets.svg',
                title='Engagement times', xlab='time (h)', ylab='counts')

    create_plot(n_retweets, root_followers, format_='o', path='sampled_conversations_graphs/retweets_vs_followers.svg',
                title='Retweets as a function of followers', xlab='followers', ylab='retweets')
    create_plot(final_rt_time, root_followers, format_='o', path='sampled_conversations_graphs/followers_vs_final_time.svg',
                title='Final engagement time', xlab='final retweet time (h)', ylab='number of followers')
    create_plot(final_rt_time, n_retweets, format_='o', path='sampled_conversations_graphs/retweets_vs_final_time.svg',
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


def reply_metrics(conv_ids: list[str]):
    """Computes reply metrics for a conversation and
    saves to folder.
    
    NOTE: Assumes that get_root_tweets() has been
    called for these conversation IDs. This is
    automatically fulfilled through calling
    get_retweets_of() with the conversation IDs
    as argument. get_root_tweets() also be called
    separately.
    
    Args:
        - conv_ids: list of conversation IDs to compute
        metrics for
        
    No return value.
    """

    n_authors_conv = []
    n_replies_conv = []
    n_deg_out_conv = []
    t_final_reply = []
    out_conv_id = []
    
    for conv_id in conv_ids: 
        
        root_path = f'root_tweets/{conv_id}_root.jsonl'
        if os.path.isfile(root_path):
            root = read_file(root_path)[0]
            out_conv_id.append(conv_id)
        else:
            continue

        conversation = read_file(f'sampled_conversations/{conv_id}')
        
        conv_dict, engagement_time, n_authors = get_conversation_dict(conversation, root)
        
        n_authors_conv.append(n_authors)
        n_replies_conv.append(len(conv_dict))
        t_final_reply.append(max(engagement_time))
        
        deg_out = 0
        for tweetID in conv_dict:
            # If root author replies to someone else [TODO: control for author]
            if conv_dict[tweetID]['author_id'] == root['author_id'] and conv_dict[tweetID]['referenced_author_id'] != root['author_id']:
                deg_out += 1
        
        n_deg_out_conv.append(deg_out)
        
    # plot figures
    t_query = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    create_plot(n_authors_conv, n_replies_conv, format_='o',
                path=f'sampled_conversations_graphs/authors_replies_{t_query}.svg',
                title='Authors and replies', xlab='number of authors', ylab='number of replies')
    create_plot(n_authors_conv, t_final_reply, format_='o',
                path=f'sampled_conversations_graphs/authors_t_final_{t_query}.svg',
                title='Authors and time for last reply', xlab='number of authors',
                ylab='time of final reply (h)')
    create_hist(n_replies_conv, bins=50,
                path=f'sampled_conversations_graphs/replies_distribution_{t_query}.svg',
                title='Replies in conversations', xlab='number of replies', ylab='counts')
    create_hist(n_authors_conv, bins=50,
                path=f'sampled_conversations_graphs/authors_distribution_{t_query}.svg',
                title='Authors in conversations', xlab='number of authors', ylab='counts')
    
    return n_authors_conv, n_replies_conv, n_deg_out_conv, t_final_reply, out_conv_id
        

def get_conversation_dict(conv_tweets: list[dict], root: dict) -> tuple[dict, list, int]:
    """Returns a dictionary containing information on
    the tweets in the conversation along with a list
    of their engagement times in hours after tweet zero.
    Dictionaries in Python >3.7 are ordered.
    
    Args:
        - conv_tweets: a list with tweet .json objects, 
        obtained from e.g., read_file()
        
    Returns:
        - conv_dict: a dictionary mapping tweet ids to tweet info
        - engagement_time: list of times in hours
        - n_authors: number of authors participating in the conversation
    """
    author_set = set()
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

        author_set.add(tw['author_id'])
        conv_dict[tw['id']] = {'author_id':      tw['author_id'],
                               'author_name':    tw['author']['name'],
                               'time_stamp':     tw['created_at'],
                               'public_metrics': tw['public_metrics'],
                               'referenced_id':  ref_id,
                               'referenced_author_id': ref_auth_id}
                             # 'text_snippet':tw['text'][0:4],
        
        time_stamps.append(tw['created_at'])
        
    engagement_time = []
    t0 = datetime.datetime.strptime(root['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')
    for ts in time_stamps:
        time = datetime.datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.000Z')
        dt = time-t0
        engagement_time.append((86400*dt.days + dt.seconds)/3600)

    # Needed if we use the results of conv. query and not root tweet time
    #for i in range(len(engagement_time)):
    #    engagement_time[i] = engagement_time[i] - engagement_time[-1]
        
    n_authors = len(author_set)
        
    return conv_dict, engagement_time, n_authors
        

def get_conversation_dict(conv_tweets: list[dict], root: dict) -> tuple[dict, list, int]:
    """Returns a dictionary containing information on
    the tweets in the conversation along with a list
    of their engagement times in hours after tweet zero.
    Dictionaries in Python >3.7 are ordered.
    
    Args:
        - conv_tweets: a list with tweet .json objects, 
        obtained from e.g., read_file()
        
    Returns:
        - conv_dict: a dictionary mapping tweet ids to tweet info
        - engagement_time: list of times in hours
        - n_authors: number of authors participating in the conversation
    
    """
    author_set = set()
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

        author_set.add(tw['author_id'])
        conv_dict[tw['id']] = {'author_id':      tw['author_id'],
                               'author_name':    tw['author']['name'],
                               'time_stamp':     tw['created_at'],
                               'public_metrics': tw['public_metrics'],
                               'referenced_id':  ref_id,
                               'referenced_author_id': ref_auth_id}
                             # 'text_snippet':tw['text'][0:4],
        
        time_stamps.append(tw['created_at'])
        
    engagement_time = []
    t0 = datetime.datetime.strptime(root['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')
    for ts in time_stamps:
        time = datetime.datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.000Z')
        dt = time-t0
        engagement_time.append((86400*dt.days + dt.seconds)/3600)

    # Needed if we use the results of conv. query and not root tweet time
    #for i in range(len(engagement_time)):
    #    engagement_time[i] = engagement_time[i] - engagement_time[-1]
        
    n_authors = len(author_set)
        
    return conv_dict, engagement_time, n_authors


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
    
    plt.figure(figsize=(9,9), clear=True)
    n, bs, _ = plt.hist(engagement_time, bins=n_bins)
    plt.title('Replies after posting')
    plt.xlabel('hours')
    plt.ylabel('replies')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

    n_norm = n/np.sum(n)
    cdf = np.cumsum(n_norm)
    cdf = np.concatenate((np.zeros(1), cdf)) # same length as bins
    plt.figure(figsize=(9,9), clear=True)
    plt.plot(bs, cdf, '-')
    plt.title('Engagement CDF')
    plt.xlabel('hours')
    if save_path:
        plt.savefig(f'{save_path[:-4]}_cdf.svg')
    else:
        plt.show()
    plt.close()

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

        conv_dict, engagement_time, n_authors = get_conversation_dict(conv_sample)

        dir_graph_sample = create_conversation_network(conv_dict, engagement_time)
        
        """
        # Room for computing metrics on the graphs!
        """
        
        nx.readwrite.gexf.write_gexf(dir_graph_sample, path=graph_file_name, encoding='utf-8')

        if plot_with_colors:
            assign_time_attributes(dir_graph_sample)

            plt.figure(figsize=(13,13), clear=True)
            vmax_ = int(engagement_time[-1]+1)
            nc = [v for k,v in nx.get_node_attributes(dir_graph_sample, 'time').items()]
            nx.draw_kamada_kawai(dir_graph_sample, with_labels=False, font_weight='bold', node_color = nc, vmin=0, vmax=vmax_, cmap = plt.cm.get_cmap('rainbow'))
            plt.savefig(f'sampled_conversations_graphs/{conv_id}_conversation_graph_users.svg')
            if show_plot:
                plt.show()
        else:
            plt.figure(figsize=(13,13), clear=True)
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