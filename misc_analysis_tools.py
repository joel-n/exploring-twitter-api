import os
import json
import logging
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from file_manip import * # file I/O
from plot_tools import * # plotting functions
import matplotlib.pyplot as plt
# import seaborn as sns # plotting tools


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