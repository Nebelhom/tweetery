#!usr/bin/env python

import csv
import time
import numpy as np
import pandas as pd
# Keys may otherwise not be properly displayed
pd.set_option("display.max_colwidth",999)
import tweepy

"""
Purpose: Download tweets from interesting feeds. classify them into interesting and not interesting

Use this to train a natural language model to classify future tweets.

Later try to classify automatically into separate categories

At the very end search large twitter space to look for fitting tweets across whole space with trained model
"""


class TweetCollector(object):
    """
    Class written to download tweets from all feeds specified and then save them in a csv file... or sql (make it sql for fun)

    currently planned format:
    twitter_feed | Tweet | Date | Interesting? (Used to train classifier) | (Optional) Keywords automatically added
    """

    def __init__(self, _feedfile='feeds.csv', _keyfile='keys.csv'):

        kd = pd.read_csv(_keyfile)
        self.ckey = kd[kd['key']=='ckey']['value'].to_string(index=False)
        self.csecret = kd[kd['key']=='csecret']['value'].to_string(index=False)
        self.atoken =  kd[kd['key']=='atoken']['value'].to_string(index=False)
        self.asecret =  kd[kd['key']=='asecret']['value'].to_string(index=False)

        self.feeds = pd.read_csv(_feedfile)

    def download(self, feed, since_id, count=200, exclude_replies=True, include_rtf=True):
        """
        Download tweets using tweepery.

        Arguments:
        feed - string - username on twitter; here comes from self.feeds dataframe
        count=200 - integer - max. 200
        since_id=None - None or string of integer
        exclude_replies=True
        - see # https://developer.twitter.com/en/docs/tweets/timelines/guides/working-with-timelines
        include_rtf=True
        - see # https://developer.twitter.com/en/docs/tweets/timelines/guides/working-with-timelines

        Returns:
        df_tweets - pandas.DataFrame - columns=['since_id','created_at','tweet']
        """

        auth = tweepy.OAuthHandler(self.ckey, self.csecret)
        auth.set_access_token(self.atoken, self.asecret)
        api = tweepy.API(auth)

        # from
        # https://gist.github.com/yanofsky/5436496

        alltweets = []

        if since_id == -1:
            new_tweets = api.user_timeline(screen_name=feed, count=count,
                exclude_replies=exclude_replies, include_rtf=include_rtf)
        else:
            new_tweets = api.user_timeline(screen_name=feed, count=count,
                since_id=since_id, exclude_replies=exclude_replies,
                include_rtf=include_rtf)

        # save most recent tweets
        alltweets.extend(new_tweets)

        #transform the tweepy tweets into a 2D array that will populate the csv 
        outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8"), feed] for tweet in alltweets]

        df_tweets = pd.DataFrame(data=outtweets, columns=['since_id', 'created_at', 'tweet', 'feed'])

        return df_tweets

    def download_all(self, feed, exclude_replies=True, include_rtf=True):
        """
        Does not quite work yet...
        """
        auth = tweepy.OAuthHandler(self.ckey, self.csecret)
        auth.set_access_token(self.atoken, self.asecret)
        api = tweepy.API(auth)

        # from
        # https://gist.github.com/yanofsky/5436496
        # and
        # https://stackoverflow.com/questions/45867934/is-it-possible-to-download-tweets-and-retweets-of-10000-users-with-tweepy-throug
        alltweets = []
        new_tweets = api.user_timeline(screen_name=feed, count=200,
            exclude_replies=exclude_replies, include_rtf=include_rtf)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        # keep grabbing tweets until there are no tweets left to grab
        while new_tweets:
            print(new_tweets[0].text.encode("utf-8"))
            print("getting tweets before {}".format(oldest))
            
            #all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = api.user_timeline(screen_name=feed, count=200,
            exclude_replies=exclude_replies, include_rtf=include_rtf,  max_id=oldest)
            
            #save most recent tweets
            alltweets.extend(new_tweets)
            
            #update the id of the oldest tweet less one
            oldest = alltweets[0].id - 1
            
            print ("...{} tweets downloaded so far".format(len(alltweets)))

        #transform the tweepy tweets into a 2D array that will populate the csv 
        outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8"), feed] for tweet in alltweets]

        df_tweets = pd.DataFrame(data=outtweets, columns=['since_id', 'created_at', 'tweet', 'feed'])

        return df_tweets

    def to_CSV(self, tweets, csvname=''):
        if csvname == '':
            csvname = '{}_tweets.csv'.format(tweets['feed'][0])
        tweets.to_csv(csvname, index=False)

if __name__ == '__main__':
    tw = TweetCollector()
    #for feed, since_id in tw.feeds.values:
    #    tweets = tw.download_all(feed, since_id)
    #    tw.to_csv('test.csv', index=False)
    tweets = tw.download_all('Nebelhom', -1)
    tw.to_CSV(tweets)