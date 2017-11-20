#!usr/bin/env python

import csv
import time
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords


# Keys may otherwise not be properly displayed
pd.set_option("display.max_colwidth",999)
import tweepy

"""
Purpose: Download tweets from interesting feeds. classify them into interesting and not interesting

Use this to train a natural language model to classify future tweets.

Later try to classify automatically into separate categories

At the very end search large twitter space to look for fitting tweets across whole space with trained model
"""

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


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
        self.columns = ['since_id', 'created_at', 'tweet', 'feed']

    def download(self):
        """
        Download and accumulate tweets from multiple feeds
        """
        alltweets = []
        for feed, since_id in self.feeds.values:
            tweets = self._download(feed, since_id)
            alltweets.append(tweets)

        return pd.concat(alltweets)


    def _download(self, feed, since_id, count=200, exclude_replies=True, include_rtf=True):
        """
        Download tweets using tweetery.

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
        outtweets = [[tweet.id_str, tweet.created_at, tweet.text, feed] for tweet in alltweets]

        df_tweets = pd.DataFrame(data=outtweets, columns=self.columns)

        return df_tweets

    def _download_all(self, feed, exclude_replies=True, include_rtf=True):
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
        outtweets = [[tweet.id_str, tweet.created_at, tweet.text, feed] for tweet in alltweets]

        df_tweets = pd.DataFrame(data=outtweets, columns=self.columns)

        return df_tweets

    def download_all(self):
        """
        Download and accumulate tweets from multiple feeds
        """
        alltweets = []
        for feed, since_id in self.feeds.values:
            tweets = self._download_all(feed, since_id)
            alltweets.append(tweets)

        return pd.concat(alltweets)

    def to_CSV(self, tweets, csvname=''):
        if csvname == '':
            csvname = 'Tweets.csv'
        tweets.to_csv(csvname, index=False)

    def to_XLS(self, tweets, xlsname=''):
        if xlsname == '':
            xlsname = 'Tweets.xlsx'
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(xlsname, engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        tweets.to_excel(writer, sheet_name='Sheet1', index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def save_feeds_csv(self, db, fname='feeds.csv'):
        """
        Saves a list of feeds and also the oldest since_id

        Steps: identify all tweet feeds in db, find the oldest of each feed in that list
        """
        feeds = db['feed'].unique()
        result = []
        for f in feeds:
            result.append([
                f,
                db[db['feed'] == f]['since_id'].max()
                ])
        new_feed = pd.DataFrame(data=result, columns=['feed', 'since_id'])
        new_feed.to_csv(fname, index=False)

    def add_remove_punct(self, tweets):
        tweets['tweet'] = tweets['tweet'].apply(text_process)

if __name__ == '__main__':
    tw = TweetCollector()
    tweets = tw.download()
    tweets['clean_tweet'] = tweets['tweet'].apply(text_process)
    tw.to_CSV(tweets)
    tw.to_XLS(tweets)
    tw.save_feeds_csv(tweets, fname='test_feeds.csv')