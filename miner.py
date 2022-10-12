import os
import logging
import datetime
import time as pytime
from file_manip import * # file I/O
from twarc import Twarc2, expansions
from analysis_tools import get_file_paths, handle_special


class miner():

    def __init__(self, academic):
        
        if academic:
            with open(f'tokens/academic_bearer_token.txt', 'r') as file:
                BEARER_TOKEN = file.read()
        else:
            with open('tokens/bearer_token.txt', 'r') as file:
                BEARER_TOKEN = file.read()

        self.client = Twarc2(bearer_token=BEARER_TOKEN)

        fetching_start = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        logging.basicConfig(filename=f'mining_logs/fetch_log_{fetching_start}.log', encoding='utf-8', level=logging.DEBUG)
        self.logger = logging.getLogger()


    def recent_search(self, query_: str, max_results_pp=100, tweet_fields_=None, user_fields_=None) -> list[dict]:
        """Query the Twitter API for data using
        the search/recent endpoint.
        
        Args:
            - query_: query string, max 512 characters
            - max_results_pp: the maximum results to return per page,
            note that all matching results will be returned
            
        Returns:
            - tweets: list of tweets in the form of json objects
        
        """
        search_results = self.client.search_recent(query=query_, max_results=max_results_pp,
                                                    tweet_fields=tweet_fields_, user_fields=user_fields_)
        tweets = []

        for page in search_results:
            result = expansions.flatten(page)
            for tweet in result:
                tweets.append(tweet)
        return tweets


    def full_archive_search(self, query_: str, max_results_pp=500, tweet_fields_=None, user_fields_=None,
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
        search_results = self.client.search_all(query=query_, max_results=max_results_pp, expansions=expansions_,
                                                tweet_fields=tweet_fields_, user_fields=user_fields_,
                                                media_fields=media_fields_, poll_fields=poll_fields_,
                                                place_fields=place_fields_)
        tweets = []

        for page in search_results:
            result = expansions.flatten(page)
            for tweet in result:
                tweets.append(tweet)
        return tweets


    def get_retweeters(self, tweet_id: str) -> list[str]:
        """Query the APi for retweeters of a specific tweet.
        
        Args:
            - tweet_id: ID of the tweet for which to fetch retweeters
            
        Returns:
            - retweeters: list of user IDs that retweeted tweet_id
        """
        retweeters = []
        results = self.client.retweeted_by(tweet_id)
        for page in results:
            res = expansions.flatten(page)
            for retweeter in res:
                retweeters.append(retweeter['id'])
        return retweeters


    def get_users_followers(self, user_ids: list[str]) -> list[dict]:
        """Returns user profiles of a list of user IDs.
        
        Args:
            - user_ids: list of user IDs
        
        Returns:
            - user_followers: dict of user followers
        """
        
        user_followers = {}
        results = self.client.user_lookup(users=user_ids, usernames=False)
        for page in results:
            res = expansions.flatten(page)
            for user in res:
                user_followers[user['id']] = user['public_metrics']['followers_count']
        return user_followers


    def get_retweeter_followers(self, conv_ids: list[str]):
        """Fetches the followers of users that retweeted a root,
        appends the dictionary to the retweet.
        Ignores files that have been processed previously or that
        already contain follower information.
        
        Args:
            - conv_ids: list of conversation IDs
            
        No return value.
        """
        for conv_id in conv_ids:
            root_path, _, retw_path = get_file_paths(conv_id)
            users = []
            if os.path.isfile(f'retweets_with_followers/{conv_id}_flws.jsonl'):
                self.logger.info(f'Retweet file {conv_id} already processed.')
                continue
            
            exists = False
            if os.path.isfile(root_path) and os.path.isfile(retw_path):
                for rt in file_generator(retw_path):
                    if 'public_metrics' in rt['author']:
                        exists = True
                    break
                if exists:
                    self.logger.info(f'Retweet file {conv_id} includes retweeter followers.')
                    continue
                
                for rt in file_generator(retw_path):
                    users.append(rt['author_id'])
                user_followers = self.get_users_followers(users)
                
                retweets = []
                for rt in file_generator(retw_path):
                    user_id = rt['author_id']
                    if user_id in user_followers:
                        rt['public_metrics'] = {'followers_count': user_followers[user_id]}
                    else:
                        self.logger.info(f'Followers of user {user_id} could not be retrieved.')
                    retweets.append(rt)
                append_objs_to_file(f'retweets_with_followers/{conv_id}_flws.jsonl', retweets)
        return


    def get_single_root(self, root_id: str) -> bool:
        """Retrieves a single (root) tweet defined by root_id.
        
        Args:
            - root_id: ID of tweet to retrieve
            
        Returns:
            - A boolean indicating whether the tweet has been
            retrieved.
        """
        if not os.path.isfile(f'root_tweets/{root_id}_root.jsonl'):
            result = self.client.tweet_lookup([root_id])
            for page in result:
                if not 'errors' in page:
                    res = expansions.flatten(page)
                    for orig_tweet in res:
                        cid = orig_tweet['conversation_id']
                        append_objs_to_file(f'root_tweets/{cid}_root.jsonl', [orig_tweet])

        return os.path.isfile(f'root_tweets/{root_id}_root.jsonl')


    def retrieve_conversations(self, sample_file: str) -> None:
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
        
        for _, tweet_dict in enumerate(sample):
            
            conv_id = tweet_dict['data']['conversation_id']
            
            # If the sampled tweet is a RT, use the retweeted tweet as root
            if 'referenced_tweets' in tweet_dict['data']:
                for ref in tweet_dict['data']['referenced_tweets']:
                    if ref['type'] == 'retweeted':
                        conv_id = ref['id']
                        break
            else:
                self.logger.info(f'Tweet {conv_id} has no referenced tweets.')

            n_results = self.__get_tweet_replies(conv_id)
            
            if n_results > 0:
                tot_conv_retrieved += 1
                self.logger.info(f'conversation {conv_id} resulted in a total of {n_results} results.')
            elif n_results == -1:
                self.logger.info(f'Root for conversation {conv_id} could not be retrieved.')
        
        t2 = pytime.time() - t1
        self.logger.info(f'Conversation retrieval took {t2} seconds. Out of {len(sample)} conversations, {tot_conv_retrieved} contained replies.')
        print('Finished retrieving {} conversations in {:.3f} minutes ({:.3f} hours)'.format(tot_conv_retrieved, t2/60, t2/3600))
        
        return


    def __get_tweet_replies(self, conv_id: str) -> int:
        """Returns and saves the replies for a single tweet from a sample
        file. Ignores tweets that have already been retrieved. Logs the
        successes and failures to the logger.
        
        Args:
            - conv_id: conversation (root tweet) ID

        Returns:
            - n_results: the number of results retrieved
        """

        if os.path.isfile(f'sampled_conversations/{conv_id}_conversation-tweets.jsonl') or os.path.isfile(f'interaction_times/reply_times/{conv_id}.txt'):
            self.logger.info(f'Conversation {conv_id} already retrieved.')
            return 0
        elif os.path.isfile(f'root_tweets/{conv_id}_root.jsonl'):
            self.logger.info(f'Conversation {conv_id} already retrieved (root exists).')
            return 0

        # Fetch root; if retrieval fails, skip the conversation.
        root_exists = self.get_single_root(conv_id)
        if not root_exists:
            self.logger.info(f'Root tweet {conv_id} could not be retrieved.')
            return -1
        
        # TODO (re-do): fix so that partial results are printed to file
        # in case of a very long conversation
        
        # Calling full archive search (rate limit: 300 requests/15 minutes)
        conv_query = f'conversation_id:{conv_id}'
        conv_tweets = self.full_archive_search(query_=conv_query, max_results_pp=100,
                                                expansions_='author_id,in_reply_to_user_id,referenced_tweets.id,referenced_tweets.id.author_id,entities.mentions.username',
                                                tweet_fields_='attachments,author_id,context_annotations,conversation_id,created_at,entities,id,public_metrics,text,referenced_tweets,reply_settings,source',
                                                user_fields_='created_at,entities,id,name,protected,public_metrics,url,username',
                                                media_fields_='type', poll_fields_='id', place_fields_='id')

        filename = f'sampled_conversations/{conv_id}_conversation-tweets.jsonl'
        n_results = len(conv_tweets)

        append_objs_to_file(filename, conv_tweets)

        return n_results


    def get_root_tweets(self, conv_ids: list[str], get_saved=False) -> list[dict]:
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
            results = self.client.tweet_lookup(filter(self.root_tweet_missing, conv_ids))
            for page in results:
                res = expansions.flatten(page)
                for orig_tweet in res:
                    cid = orig_tweet['conversation_id']
                    append_objs_to_file(f'root_tweets/{cid}_root.jsonl', [orig_tweet])
                    n_res += 1
            
            print(f'Retrieved {n_res} new roots.')
            
            # Retrieve all saved roots
            roots, retrieved_conv_ids = self.get_root_tweets(conv_ids, True)
            
        return roots, retrieved_conv_ids


    def root_tweet_missing(self, conv_id):
        """Returns True if root tweet with ID == conv_id is missing."""
        return not os.path.isfile(f'root_tweets/{conv_id}_root.jsonl')


    def get_all_retweets(self, conv_ids: list[str]) -> None:
        """Get retweets of conversation roots and stores them in a .jsonl-file.
        Will ignore the conversation IDs that already is associated with a 
        retweet file. Efficient query for all RTs containing the text, and is
        an RT of the root author. In practice, not all retweets can be fetched
        due to some profiles being protected.
            
        Args:
            - conv_ids: list of conversation IDs to fetch retweets for
            
        No return value.    
        """

        roots, retrieved_conv_ids = self.get_root_tweets(conv_ids, get_saved=True)
        
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
            if os.path.isfile(f'retweets/{retrieved_conv_ids[i]}.jsonl') or os.path.isfile(f'interaction_times/retweet_times/{retrieved_conv_ids[i]}.txt'):
                roots_skipped += 1
                self.logger.info(f'Skipped retweets of conversation {retrieved_conv_ids[i]} as they were already retrieved.')
                continue

            max_results_pp = 500
            n_pages, tot_results, num_rts, api_mismatch = self.__retweet_query(retrieved_conv_ids[i], max_results_pp)

            # Warn if there are many retweets according to root, but none could be retrieved
            if api_mismatch:
                self.logger.info(f'Retweets to conversation {retrieved_conv_ids[i]} resulted in too many matches and was aborted after {n_pages} pages of results.')
                self.logger.info(f'Expected {int(1 + num_rts/max_results_pp)} pages for {num_rts} retweets. Retrieved {tot_results} retweets before aborting.')
                continue

            # already appended in loop: append_objs_to_file(f'retweets/{retrieved_conv_ids[i]}.jsonl', retweets)
            self.logger.info(f'Retrieved {tot_results} (of {num_rts}) retweets to conversation {retrieved_conv_ids[i]}.')
            
        t2 = pytime.time() - t1
        self.logger.info(f'Retweets retrieval took {t2} seconds.')
        print('Retweets retrieval took {:.3f} minutes ({:.3f} hours); '.format(t2/60, t2/3600))
        print('Skipped {} roots for which RTs are already retrieved, and {} roots which contained no searchable text.'.format(roots_skipped, no_text_roots))
        
        return


    def __get_retweets(self, conv_id: str):
        """Retrieves all available retweets of a single conversation. Root dictionary
        must exist as file.
        
        Args:
            - conv_id: conversation ID
        
        Returns:
            - An integer. If retrieval failed, -1, else the number
            of retrieved retweets.
        """

        if os.path.isfile(f'retweets/{conv_id}.jsonl') or os.path.isfile(f'interaction_times/retweet_times/{conv_id}.txt'):
            self.logger.info(f'Retweets of conversation {conv_id} are already retrieved.')
            return 0

        max_results_pp = 500
        n_pages, tot_results, num_rts, api_mismatch = self.__retweet_query(conv_id, max_results_pp)

        # Warn if there are many retweets according to root, but none could be retrieved
        if api_mismatch:
            self.logger.info(f'Retweets to conversation {conv_id} resulted in too many matches and was aborted after {n_pages} pages of results.')
            self.logger.info(f'Expected {int(1 + num_rts/max_results_pp)} pages for {num_rts} retweets. Retrieved {tot_results} retweets before aborting.')
            return -1

        # already appended in loop: append_objs_to_file(f'retweets/{retrieved_conv_ids[i]}.jsonl', retweets)
        self.logger.info(f'Retrieved {tot_results} (of {num_rts}) retweets to conversation {conv_id}.')
        
        return tot_results


    def __retweet_query(self, conv_id: str, max_results_pp: int):
        """Makes a query for retweets of a conversation and stores the results.
        
        Args:
            - conv_id: conversation ID
            - max_results_pp: maximum results per page of query

        Returns:
            - n_pages: number of results pages retrieved
            - tot_results: total number of retrieved retweets
            - n_rts: retweets according to the API
            - api_mismatch: flag indicating early abortion of retrieval
        """
        root_dict = read_file(f'root_tweets/{conv_id}_root.jsonl')[0]

        # DECISION: Heuristic to get rid of links in tweets
        root_text = root_dict['text'].split('https://t.co/')[0]
        # DECISION: We could also take root_text = ''.join(root['text'].split('https://t.co/')[:-1])
        # to remove the last link (alt. use rfind())
        
        # Handle quotation marks
        #if root_text.find('"') != -1 or root_text.find('&amp;') =! -1 or root_text.find('&gt;') =! -1 or root_text.find('&lt;'):
        if '"' in root_text or '&amp;' in root_text or '&gt;' in root_text or '&lt;' in root_text:
            root_text = handle_special(root_text)
            print(f'Handled quotation marks in conversation {conv_id}; text:', root_text)
        elif not root_text:
            write_text('skipped_retweets.txt', 'skipping conversation {} (text: {})'.format(conv_id, root_dict['text']))
            return -1, -1, root_dict['public_metrics']['retweet_count'], False
        
        root_author = root_dict['author_id']
        print(conv_id, ',', root_dict['public_metrics']['retweet_count'], 'retweets')
        print(root_text)
                
        # TODO: Modify the expansion fields as well? See twitter or Postman WS for allowed values
        # TODO: Use try catch to avoid breaking HTTPerrors?

        # TODO: query also on time if we search for tweets that yield too many results
        search_results = self.client.search_all(query=f'retweets_of:{root_author} "{root_text}"', max_results=max_results_pp,
                                                expansions='author_id,in_reply_to_user_id,referenced_tweets.id,referenced_tweets.id.author_id',
                                                tweet_fields='id,author_id,referenced_tweets,created_at,source', user_fields='id,public_metrics',
                                                media_fields='type', poll_fields='id', place_fields='id')
        
        num_rts = root_dict['public_metrics']['retweet_count']
        elbow_space = 2 if num_rts > 10000 else 4
        n_expected_pages = int(elbow_space * (2 + num_rts/max_results_pp)) # allow double the expected pages

        n_pages, tot_results = 0, 0
        api_mismatch = False
        for page in search_results:
            retweets = []
            result = expansions.flatten(page)
            for retweet in result:
                for t in retweet['referenced_tweets']:
                    if t['id'] == conv_id:
                        retweets.append(retweet)
            tot_results += len(retweets)
            append_objs_to_file(f'retweets/{conv_id}.jsonl', retweets)
            n_pages += 1
            if n_pages > n_expected_pages:
                api_mismatch = True
                break
        
        # Add empty file if there were no results (does not include the case of many hits but no relevant results)
        if n_pages == 0:
            append_objs_to_file(f'retweets/{conv_id}.jsonl', [])

        return n_pages, tot_results, num_rts, api_mismatch


    def retrieve_quotes(self, conv_ids: list[str]) -> None:
        """Retrieve quotes of a conversation using the quotes
        endpoint. Stores the quotes in the quotes folder.
        Quote queries are limited to 75 per 15 minutes, and
        100 results per page.
        
        Args:
            - conv_ids: list of conversation IDs for which quotes are
            to be retrieved.
        
        No return value.
        """
        n_conv = len(conv_ids)
        print('Retrieving quotes for {} tweets.\
        With 300 queries per hour this should take {:.3f} minutes ({:.3f} hours)'.format(n_conv, n_conv/5, n_conv/300))
        t1 = pytime.time()
        tot_quotes_retrieved = 0
        
        for conv_id in conv_ids:
            n_results = self.__get_quotes(conv_id, check_root=True)            
            
            if n_results > 0:
                tot_quotes_retrieved += 1
        
        t2 = pytime.time() - t1
        self.logger.info(f'Conversation retrieval took {t2} seconds. Out of {n_conv} conversations, {tot_quotes_retrieved} were quoted.')
        print('Finished retrieving quotes for {} conversations in {:.3f} minutes ({:.3f} hours)'.format(tot_quotes_retrieved, t2/60, t2/3600))        
        return


    def __get_quotes(self, conv_id: str, check_root=True):
        """Retrieves all available quotes of a conversation.
        
        Args:
            - conv_id: conversation ID
            - check_root: if True, will check whether the root
            exists and/or can be retrieved
        """

        quote_path = f'quotes/{conv_id}_quotes.jsonl'
                        
        if os.path.isfile(quote_path) or os.path.isfile(f'interaction_times/quote_times/{conv_id}.txt'):
            self.logger.info(f'Quotes for {conv_id} already retrieved.')
            return 0
        elif check_root and not self.get_single_root(conv_id):
            self.logger.info(f'Cannot get root for tweet {conv_id}.')
            return -1

        root = read_file(f'root_tweets/{conv_id}_root.jsonl')[0]
        if root['public_metrics']['quote_count'] == 0:
            append_objs_to_file(quote_path, [])
            self.logger.info(f'Tweet {conv_id} has no quote tweets.')
            return 0
        
        exp = 'author_id,in_reply_to_user_id,referenced_tweets.id,referenced_tweets.id.author_id,entities.mentions.username'
        tw_fields = 'attachments,author_id,context_annotations,conversation_id,created_at,entities,id,public_metrics,text,referenced_tweets,reply_settings'
        usr_fields = 'created_at,entities,id,name,protected,public_metrics,url,username'
        quote_results = self.client.quotes(conv_id, expansions=exp, tweet_fields=tw_fields, user_fields=usr_fields, max_results=100)
        
        n_results = 0
        for page in quote_results:
            result = expansions.flatten(page)
            append_objs_to_file(quote_path, result)
            n_results += len(result)
            
        self.logger.info(f'Quote query on {conv_id} resulted in a total of {n_results} tweets.')
        if n_results == 0:
            append_objs_to_file(quote_path, [])
            
        return n_results


    def get_single_tweet_reply_retweet_quote(self, conv_id: str) -> None:
        """Retrieves replies, retweets and quotes for a single conversation ID.
        
        Args:
            - conv_id: conversation ID
        """

        # Retrieve conversation replies, will not fetch replies if they already exist
        n_results = self.__get_tweet_replies(conv_id)
        if n_results > 0:
            self.logger.info(f'Retrieved {n_results} replies for conversation {conv_id}.')
        elif n_results == -1:
            self.logger.info(f'Root for conversation {conv_id} could not be retrieved.')
            return

        # Get retweets
        n_rt_results = self.__get_retweets(conv_id)
        if n_rt_results > 0:
            self.logger.info(f'Retrieved {n_rt_results} retweets for conversation {conv_id}.')
        elif n_rt_results == -1:
            self.logger.info(f'Failed to retrieve retweets for conversation {conv_id} (no text or API mismatch).')
        
        # Get quotes
        n_q_results = self.__get_quotes(conv_id, check_root=False)
        if n_q_results > 0:
            self.logger.info(f'Retrieved {n_rt_results} quotes for conversation {conv_id}.')
        elif n_q_results == -1:
            self.logger.info(f'Failed to retrieve retweets for conversation {conv_id}.')

        return


    def retrieve_interactions(self, conv_ids: list[str]) -> None:
        """Retrieves all interactions of the conversations in the provided list.
        
        Args:
            - conv_ids: list of conversation IDs
        """
        for conv_id in conv_ids:
            self.get_single_tweet_reply_retweet_quote(conv_id)
            self.logger.info(f'Finished retrieving interactions for conversation {conv_id}.')
        return


    def retrieve_interactions_from_sample(self, sample_file: str) -> None:
        """Retrieves all interactions of the conversations in the provided in
        the sample stream file.
        
        Args:
            - sample_file: path to sample stream file
        """

        sample = read_file(sample_file)
        print('Retrieving interactions for {} tweets.'.format(len(sample)))
        t1 = pytime.time()
        tot_conv_retrieved = 0
        
        for _, tweet_dict in enumerate(sample):
            
            conv_id = tweet_dict['data']['conversation_id']
            if os.path.isfile(f'root_tweets/{conv_id}_root.jsonl'):
                continue
            
            # If the sampled tweet is a RT, use the retweeted tweet as root
            if 'referenced_tweets' in tweet_dict['data']:
                for ref in tweet_dict['data']['referenced_tweets']:
                    if ref['type'] == 'retweeted':
                        conv_id = ref['id']
                        break
            else:
                self.logger.info(f'Tweet {conv_id} has no referenced tweets.')

            self.get_single_tweet_reply_retweet_quote(conv_id)
            self.logger.info(f'Finished retrieving interactions for conversation {conv_id}.')
            tot_conv_retrieved += 1
        
        t2 = pytime.time() - t1
        self.logger.info(f'Conversation retrieval took {t2} seconds. Out of {len(sample)} conversations, {tot_conv_retrieved} contained replies.')
        print('Finished retrieving {} conversations in {:.3f} minutes ({:.3f} hours)'.format(tot_conv_retrieved, t2/60, t2/3600))
        return


    def get_users(self, username, user_fields_='created_at,id,name,username,public_metrics'):
        """Returns a hydrated list of user profiles
        from the list of IDs provided.
        
        Args:
            - username: list of usernames for the accounts to look up.
            - user_fields_: optional string with query attributes
            
        Returns:
            - userlist: list of user profiles with the attributes
            in user_fields_.
        """
        
        users = self.client.user_lookup(username, usernames=True, expansions=None, tweet_fields=None, user_fields=user_fields_)
        userlist = []
        for page in users:
            result = expansions.flatten(page)
            for usr in result:
                userlist.append(usr)
        return userlist


    def retrieve_followers(self, user_ids: list[str], folder_name='infl_follower_ids'):
        """Saves the user ID of the followers of user_id to file.

        Args:
            - user_ids: list of IDs of accounts for which to
            retrieve followers.
        
        No return value.
        """
        for user_id in user_ids:
            file_name = f'{folder_name}/{user_id}_followers.txt'
            if os.path.isfile(file_name):
                self.logger.info(f'Followers for {user_id} already retrieved.')
                continue

            followers = self.client.followers(user=user_id)
            
            for page in followers:
                if not 'data' in page:
                    append_ids_to_file(file_name, [])
                    self.logger.info(f'Error when retrieving followers of {user_id}.\n{page}')
                    break
                result = expansions.flatten(page)
                append_ids_to_file(file_name, result)
        return


    def get_followers(self, user_, load=True):
        """OUTDATED FUNCTION, USE retrieve_followers() INSTEAD.
        Saves and (optionally) returns a list of user profiles and IDs
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
        followers = self.client.followers(user=user_, user_fields='created_at,id,name,username,public_metrics')
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


    def get_following(self, user_, load=True):
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
        
        following = self.client.following(user=user_, user_fields='created_at,id,name,username,public_metrics')
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


    def root_tweet_missing(self, conv_id):
        """Returns True if root tweet with ID == conv_id is missing."""
        return not os.path.isfile(f'root_tweets/{conv_id}_root.jsonl')