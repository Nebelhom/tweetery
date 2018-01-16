#!usr/bin/python

import datetime
import pandas as pd
import numpy as np

from collector import TweetCollector
from classifier import TweetClassifier

from config import env_args

TODAY = datetime.date.today()

if __name__ == '__main__':
    # Instantiate TweetCollector
    tw = TweetCollector(feedfile=env_args['feedfile'])
    # Download the tweets based on feedfile
    tw.download_tweets()
    # Save tweets in CSV or XLS file
    tw.to_CSV(csvname=env_args['tweetcsv'], overwrite=True,
              extend_existing=False)
    tw.to_XLS(xlsname=env_args['tweetxls'], overwrite=True,
              extend_existing=True)
    # Update the feedfile
    tw.update_feeds_csv(fname=env_args['feedfile'])

    # Get tweets to be classified
    clf = tw.get_tweets()

    # Load training data. If Classifier is already saved, there is no need
    df = pd.read_excel(env_args['trainfile'])

    # Instantiate TweetClassifier
    tc = TweetClassifier(clf, train=df)

    # Save the trained classifier
    tc.save_classifier()

    # Predict the downloaded tweets
    tc.predict()

    # Save them as report
    tc.save_as_doc(fname=env_args['reportdoc'])
    #tc.save_as_txt(fname=env_args['reporttxt'])
    tc.save_as_email(subject='Twitter Report {}'.format(TODAY),
                     fname=env_args['reporteml'])

    # Extend the existing training Data by the newly classified ones
    # It may be necessary to change some things manually if classification
    # was wrong
    tweets = tc.get_prediction().drop('probability', axis=1)
    tw.to_XLS(xlsname=env_args['trainfile'], tweets=tweets, overwrite=True,
              extend_existing=True)
