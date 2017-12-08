#!usr/bin/env python

import csv
import time
import numpy as np
import pandas as pd
import os.path
import string
from nltk.corpus import stopwords

# https://pypi.python.org/pypi/emoji/
import emoji


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

    def __init__(self, feedfile='feeds.csv', keyfile='keys.csv'):

        kd = pd.read_csv(keyfile)
        self.ckey = kd[kd['key']=='ckey']['value'].to_string(index=False)
        self.csecret = kd[kd['key']=='csecret']['value'].to_string(index=False)
        self.atoken =  kd[kd['key']=='atoken']['value'].to_string(index=False)
        self.asecret =  kd[kd['key']=='asecret']['value'].to_string(index=False)

        self.feeds = pd.read_csv(feedfile)
        self.columns = ['since_id', 'created_at', 'tweet', 'feed']

        self.tweets = self.get_tweets()

    def get_tweets(self):
        """
        Download and accumulate tweets from multiple feeds
        """
        alltweets = []
        for feed, since_id in self.feeds.values:
            tweets = self._download(feed, since_id)
            alltweets.append(tweets)

        return pd.concat(alltweets)

    def clean_tweet_col(self, new_col="clean_tweet"):
        """
        Creates a new column in the pandas dataframe with
        precursor to bag of words using text_process method
        """
        self.tweets[new_col] = self.tweets['tweet'].apply(self.text_process)

    def text_process(self, mess):
        """
        Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords
        3. Returns a list of the cleaned text
        """
        # Check characters to see if they are in punctuation
        nopunc = [char for char in mess if char not in string.punctuation + "–"]

        # Check if char is an emoji
        noemoji = [char for char in nopunc if char not in emoji.UNICODE_EMOJI.keys()]

        # Join the characters again to form the string.
        nopunc = ''.join(noemoji)

        # Now just remove any stopwords
        return [word for word in nopunc.split() if word.lower() not in stopwords.words('english') if 'http' not in word.lower()]


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
                exclude_replies=exclude_replies, include_rtf=include_rtf,
                tweet_mode='extended')
        else:
            new_tweets = api.user_timeline(screen_name=feed, count=count,
                since_id=since_id, exclude_replies=exclude_replies,
                include_rtf=include_rtf, tweet_mode='extended')

        # save most recent tweets
        alltweets.extend(new_tweets)

        #transform the tweepy tweets into a 2D array that will populate the csv

        outtweets = [[tweet.id_str, tweet.created_at,
                      self.get_tweet_full_text(tweet), feed]
                     for tweet in alltweets]

        df_tweets = pd.DataFrame(data=outtweets, columns=self.columns)

        return df_tweets

    def check_ending(self, fname, ending):
        """
        Checks for correct ending.
        """
        fend = fname.split('.')[-1]
        if fend.lower() != ending.lower():
            return False
        return True

    def new_fname_if_need(self, fname):
        """
        Checks if file exists and proposes new fname if needed.

        Relies on the fname having a .xxx ending
        """
        checking = True
        i = 1
        while checking:
            if os.path.exists(fname):
                new = fname.split('.')
                new[-2] = new[-2] + str(i)
                if os.path.exists('.'.join(new)):
                    i += 1
                else:
                    return '.'.join(new)
            else:
                return fname

    def to_CSV(self, csvname='', extend_existing=False, overwrite=False):
        """
        Writes self.tweets to CSV format.

        Keyword arguments:
        csvname         -- string -- path or filename of the CSV file.
                        If it already exists, then append number until new
                        filename created.
        extend_existing -- Boolean -- If True, extends existing csv file
        overwrite       -- Boolean -- If True, overwrites existing csv file
                        extend_existing supersedes overwrite
        """

        if csvname == '':
            csvname = 'Tweets.csv'

        # Add file ending if left out
        if not self.check_ending(csvname, 'csv'):
            csvname = csvname + '.csv'

        # Writes to a new file
        if not overwrite and not extend_existing:
            csvname = self.new_fname_if_need(csvname)
            print("Saving to {}".format(csvname))
            self.tweets.to_csv(csvname, index=False)

        # Extends an existing file if valid path given
        # Extend supersedes overwrite
        elif extend_existing:
            try:
                old_tweets = pd.read_csv(csvname)
                if old_tweets.columns.equals(self.tweets.columns):
                    new_tweets = pd.concat([self.tweets, old_tweets])
                    print('Extending tweets in {}...'.format(csvname))
                    new_tweets.to_csv(csvname, index=False)
                else:
                    print('Files do not have the same DataFrame columns. '
                          'Saving to different filename instead.')
                    self.to_CSV(csvname)

            except FileNotFoundError:
                print('File could not be found. Creating new file with '
                      'filename {}...'.format(csvname))
                self.tweets.to_csv(csvname, index=False)
            except:
                raise

        # Overwrites an existing file if needed
        else:
            print("Saving to {}".format(csvname))
            self.tweets.to_csv(csvname, index=False)

    def save_xls(self, xlsname, tweets):
        """
        Saves self.tweets dataframe to xls format.

        Auxiliary function for to_XLS.
        """
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(xlsname, engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        tweets.to_excel(writer, sheet_name='Sheet1', index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def to_XLS(self, xlsname='', extend_existing=False, overwrite=False):
        """
        Writes self.tweets to XLS or XLSX format.

        Keyword arguments:
        xlsname         -- string -- path or filename of the XLS file.
                        If it already exists, then append number until new
                        filename created.
        extend_existing -- Boolean -- If True, extends existing csv file
        overwrite       -- Boolean -- If True, overwrites existing csv file
                        extend_existing supersedes overwrite
        """
        if xlsname == '':
            xlsname = 'Tweets.xlsx'

        # Add file ending if left out
        if not self.check_ending(xlsname, 'xlsx') and \
                not self.check_ending(xlsname, 'xls'):
            xlsname = xlsname + '.xlsx'

        # Writes to a new file
        if not overwrite and not extend_existing:
            xlsname = self.new_fname_if_need(xlsname)
            print("Saving to {}".format(xlsname))
            self.save_xls(xlsname, self.tweets)

        # Extends an existing file if valid path given
        # Extend supersedes overwrite
        elif extend_existing:
            try:
                old_tweets = pd.read_excel(xlsname)
                if old_tweets.columns.equals(self.tweets.columns):
                    new_tweets = pd.concat([self.tweets, old_tweets])
                    print('Extending tweets in {}...'.format(xlsname))
                    self.save_xls(xlsname, new_tweets)

                else:
                    print('Files do not have the same DataFrame columns. '
                          'Saving to different filename instead.')
                    self.to_XLS(xlsname)

            except FileNotFoundError:
                print('File could not be found. Creating new file with '
                      'filename {}...'.format(xlsname))
                self.save_xls(xlsname, self.tweets)
            except:
                raise

        # Overwrites an existing file if needed
        else:
            print("Saving to {}".format(xlsname))
            self.save_xls(xlsname, self.tweets)

    def get_tweet_full_text(self, tweet):
        """
        Finds full text of the tweet. No matter if normal length, extended or
        retweeted.
        """
        full_text = ""

        if 'retweeted_status' in tweet._json:
            full_text = tweet._json['retweeted_status']['full_text']

        elif "extended_tweet" in tweet._json and "full_text" in tweet._json["extended_tweet"]:
            full_text = tweet._json["extended_tweet"]["full_text"]

            if "display_text_range" in tweet._json["extended_tweet"]:
                beginIndex = tweet._json["extended_tweet"]["display_text_range"][0]
                endIndex   = tweet._json["extended_tweet"]["display_text_range"][1]
                full_text = full_text[beginIndex:endIndex]

        elif "full_text" in tweet._json:
            full_text = tweet._json["full_text"]

            if "display_text_range" in tweet._json:
                beginIndex = tweet._json["display_text_range"][0]
                endIndex   = tweet._json["display_text_range"][1]
                full_text = full_text[beginIndex:endIndex]

        elif "text" in tweet._json:
            text = tweet._json["text"]
            full_text = text
        else:
            text = tweet.text
            full_text = text

        return full_text

    def save_feeds_csv(self, fname='feeds.csv'):
        """
        Saves/Updates a list of feeds and also the oldest since_id
        """
        feeds = self.tweets['feed'].unique()
        for f in feeds:
            since_id = self.tweets[self.tweets['feed'] == f]['since_id'].max()
            self.feeds.loc[self.feeds['feed'] == f, 'since_id'] = since_id
        self.feeds.to_csv(fname, index=False)

if __name__ == '__main__':
    tw = TweetCollector(feedfile='example_feeds.csv')
    # Adds another column called clean_tweets
    #tw.clean_tweet_col()
    tw.to_CSV(csvname='bla', overwrite=True, extend_existing=True)
    tw.to_XLS(xlsname='Competitor_Tweets', overwrite=True, extend_existing=True)
    #tw.save_feeds_csv(fname='example_feeds.csv')