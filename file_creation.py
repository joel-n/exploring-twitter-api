import os
from numpy import array as nparr
from file_manip import file_generator, append_objs_to_file, read_file, write_text
from analysis_tools import get_file_paths, compute_relative_time


def make_infl_id_file(conv_id: str, threshold: float, threshold_percentile: float) -> None:
    """Writes influencer IDs and follower count to file, given a threshold
    computed in advance. Does not create a file in case no influencers are
    found.

    Args:
        - conv_id: Conversation ID
        - threshold: threshold on followers to count as an influencer
        - threshold_percentile: Percentile (divided by 100) of local/global follower
        distribution used to compute the threshold. Will be part of the file name.
    """
    threshold_percentile = str(threshold_percentile).replace('.', '')
    infl_id_path = f'infl_ids/{conv_id}_{threshold_percentile}.txt'
    infl_nflw_path = f'infl_nflws/{conv_id}_nflws_{threshold_percentile}.txt'
    
    if os.path.isfile(infl_id_path):
        return
    
    infl_attr = set() # set of influencer ID and number of followers, hopefully the combo is unique!
    paths = [*get_file_paths(conv_id)] # TODO: option to read files from disk E:/
    for path in paths:
        for tweet in file_generator(path):
            if tweet['author']['public_metrics']['followers_count'] > threshold:
                infl_attr.add((int(tweet['author']['id']), tweet['author']['public_metrics']['followers_count']))
                #infl_attr[tweet['author']['id']] = tweet['author']['public_metrics']['followers_count']
    
    if len(infl_attr) == 0:
        return
    
    append_objs_to_file(infl_id_path, [attr[0] for attr in infl_attr])
    append_objs_to_file(infl_nflw_path, [attr[1] for attr in infl_attr])
    return


def make_infl_flwr_matching(conv_ids: list[str]) -> None:
    """Makes files containing interactions of influencers and influencer followers
    for a set of conversations. Stored in the folder matched_infl. The output is
    a .csv-file with influencer ID (implying that the user follows this user), user
    ID, interaction time, and interaction type as columns. Interactions from influencers
    are also added, where the user ID is simply the influencer user ID.
    Note that all reply, retweet and quote files of the conversation must exist,
    and there must be a list of follower IDs for each influencer in the conversation.

    Args:
        - conv_ids: a list of conversation IDs
    """
    already_matched, missing_data = 0, 0

    for conv_id in conv_ids:
    
        s_path = f'matched_infl/{conv_id}_user_infl_interactions.txt'

        if os.path.isfile(s_path):
            already_matched += 1
            continue
            
        # Read influencer files and see where the influencers interact with the conversation.
        infl_ids = read_file(f'infl_ids/{conv_id}_095.txt')
        existence = nparr([os.path.isfile(f'infl_follower_ids/{i_id}_followers.txt') for i_id in infl_ids])
        
        if not existence.all():
            missing_data += 1
            continue
        
        root = read_file(f'root_tweets/{conv_id}_root.jsonl')[0]
        root_time = root['created_at']
        interaction = {}

        # Read interaction files of time stamps and the IDs of users interacting
        file_paths = [
            f'sampled_conversations/{conv_id}_conversation-tweets.jsonl',
            f'retweets/{conv_id}.jsonl',
            f'quotes/{conv_id}_quotes.jsonl'
        ]

        for path in file_paths:
            interaction_type = path[0]
            for tw in file_generator(path):
                time, ID,  = tw['created_at'], tw['author_id']
                t = compute_relative_time(root_time, time)
                
                if not ID in interaction:
                    interaction[ID] = []
                interaction[ID].append((t, interaction_type))
        
        # Read list of the followers of the influencers
        # Store matching infl.-follower IDs, time difference and interaction types.
        for i_id in infl_ids:
            infl_follower_file, infl_id = f'infl_follower_ids/{i_id}_followers.txt', str(i_id)
            
            if infl_id in interaction:
                infl_interactions = interaction[infl_id]
                for ii in infl_interactions:
                    dat = '{},{},{},{}'.format(infl_id, infl_id, ii[0], ii[1])
                    write_text(s_path, dat)

            for u_id in file_generator(infl_follower_file):
                user_id = str(u_id)
                
                if user_id in interaction:
                    user_interactions = interaction[user_id]
                    for ui in user_interactions:
                        dat = '{},{},{},{}'.format(infl_id, user_id, ui[0], ui[1])
                        write_text(s_path, dat)
    
    print(f'Out of {len(conv_ids)} conversations, {already_matched} were already processed, and influencer follower IDs were missing for {missing_data}.')
    return