#!usr/bin/python

import pandas as pd
import numpy as np

from collector import TweetCollector
from classifier import Tweet_Classifier


if __name__ == '__main__':
    # Instantiate TweetCollector
    tw = TweetCollector(feedfile='user_data/example_feeds.csv')
    # Download the tweets based on feedfile
    tw.download_tweets()
    # Save tweets in CSV or XLS file
    tw.to_CSV(csvname='example_tweets.csv', overwrite=True,
              extend_existing=False)
    tw.to_XLS(xlsname='example_tweets.xlsx', overwrite=True,
              extend_existing=True)
    # Update the feedfile
    tw.update_feeds_csv(fname='user_data/example_feeds.csv')

    # Get tweets to be classified
    clf = tw.get_tweets()
    # Load training data. If Classifier is already saved, there is no need
    df = pd.read_excel('Training Data.xlsx')
    # Instantiate Tweet_Classifier
    tc = Tweet_Classifier(clf, train=df)
    # Save the trained classifier
    tc.save_classifier()
    # Predict the downloaded tweets
    tc.predict()
    # Save them as report
    tc.save_as_doc()
    tc.save_as_txt()

    # Extend the existing training Data by the newly classified ones
    # It may be necessary to change some things manually if classification
    # was wrong
    tweets = tc.get_prediction().drop('probability', axis=1)
    tw.to_XLS(xlsname='Training Data.xlsx', tweets=tweets, overwrite=True,
              extend_existing=True)
