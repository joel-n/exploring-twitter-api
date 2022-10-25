from os.path import isfile
from numpy import array as nparr
from numpy import quantile
from file_manip import file_generator, append_objs_to_file, read_file, write_text
from analysis_tools import get_file_paths, compute_relative_time, load_conv_followers


def make_unique_flw_file(conv_ids: list[str], read_from_external_disk = False):
    """Creates a file of the follower counts for unique users in
    the conversations provided in the list. Will not write files
    for conversations for which such files exist, or where data
    is missing.

    Args:
        - conv_ids: list of conversation IDs
        - read_from_external_disk: If True, reads conversation
        data from the external disk file structure.

    """
    convs_proc, follower_file_existed, missing_data = 0, 0, 0

    for c_id in conv_ids:

        comp_follower_file = f'follower_distr/unique_followers/{c_id}_unique_follower_counts.txt'
        if isfile(comp_follower_file):
            follower_file_existed += 1
            continue

        if read_from_external_disk:
            conv_path = f'E:/tweets/conversations/{c_id}_conversation-tweets.jsonl'
            retw_path = f'E:/tweets/retweets/{c_id}.jsonl'
            quote_path = f'E:/tweets/quotes/{c_id}_quotes.jsonl'
        else:
            conv_path = f'sampled_conversations/{c_id}_conversation-tweets.jsonl'
            retw_path = f'retweets/{c_id}.jsonl'
            quote_path = f'quotes/{c_id}_quotes.jsonl'
        
        if not isfile(conv_path) or not isfile(retw_path) or not isfile(quote_path):
            missing_data += 1
            continue

        root = read_file(f'root_tweets/{c_id}_root.jsonl')[0]
        root_flw = int(root['author']['public_metrics']['followers_count'])
        root_auth_id = root['author_id']
        
        users = {root_auth_id: root_flw}
        for re in file_generator(conv_path):
            users[re['author_id']] = re['author']['public_metrics']['followers_count']
        for rt in file_generator(retw_path):
            users[rt['author_id']] = rt['author']['public_metrics']['followers_count']
        for q in file_generator(quote_path):
            users[q['author_id']] = q['author']['public_metrics']['followers_count']

        convs_proc += 1

        with open(comp_follower_file, 'a+') as filehandle:
            for u_id, flw in users.items():
                filehandle.write('%d\n' % flw)

    print(f'Processed {convs_proc} conversations. {follower_file_existed} already existed, and in {missing_data} data files are missing.')
    return


def make_infl_id_file_ne(conv_ids: list[str], threshold_percentile: float) -> None:
    """Writes influencer IDs and follower count to file. Does
    not create a file in case no influencers are found, or in
    the case that such a file already exists, or data is either
    sparse (<50 interactions) or missing.

    Args:
        - conv_ids: list of conversation IDs
        - threshold_percentile: Percentile (divided by 100) of local/global follower
        distribution used to compute the threshold. Will be part of the file name.
    """
    processed, already_exists, missing_data, too_few = 0, 0, 0, 0

    for conv_id in conv_ids:
        threshold_percentile = str(threshold_percentile).replace('.', '')
        infl_id_path = f'infl_ids/{conv_id}_{threshold_percentile}.txt'
        infl_nflw_path = f'infl_nflws/{conv_id}_nflws_{threshold_percentile}.txt'
        
        if isfile(infl_id_path):
            already_exists += 1
            continue
        
        infl_attr = dict() # set of influencer ID and number of followers, hopefully the combo is unique!
        paths = [*get_file_paths(conv_id)] # TODO: option to read files from disk E:/
        
        follower_dist_path = f'follower_distr/unique_followers/{conv_id}_unique_follower_counts.txt'
        paths_exist = [isfile(p) for p in paths]
        
        if not all(paths_exist) or not isfile(follower_dist_path):
            missing_data += 1
            continue

        unique_follow_counts = nparr(load_conv_followers(follower_dist_path))
        pval = max(0.95, 1-5/len(unique_follow_counts))
        threshold = quantile(unique_follow_counts, q=pval)

        counter = 0
        for path in paths[1:]: # Skip root tweet
            for tweet in file_generator(path):
                counter += 1
                if tweet['author']['public_metrics']['followers_count'] > threshold:
                    infl_attr[tweet['author']['id']] = tweet['author']['public_metrics']['followers_count']
        
        if counter < 50:
            too_few += 1
            continue

        i_ids, i_nflws = [], []
        
        for id_,nflw_ in infl_attr.items():
            i_ids.append(int(id_))
            i_nflws.append(nflw_)
        
        append_objs_to_file(infl_id_path, i_ids)
        append_objs_to_file(infl_nflw_path, i_nflws)
        processed += 1

    print(f'Created files for {processed} conversations. Files for {already_exists} conversations already exists.')
    print(f'In {missing_data} cases data was missing, and in {too_few} cases there were too few interactions.')
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

        if isfile(s_path):
            already_matched += 1
            continue
            
        # Read influencer files and see where the influencers interact with the conversation.
        infl_ids = read_file(f'infl_ids/{conv_id}_095.txt')
        existence = nparr([isfile(f'infl_follower_ids/{i_id}_followers.txt') for i_id in infl_ids])
        
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
        # Store matching infl. follower ID pairs, time difference and interaction types.
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
    
    print(f'Out of {len(conv_ids)} conversations, {already_matched} were already processed, and influencer follower IDs were missing for {missing_data}. Files were created for {len(conv_ids)-already_matched-missing_data} conversations.')
    return