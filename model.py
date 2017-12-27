#!usr/bin/python

import pandas as pd
import numpy as np

import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# https://pypi.python.org/pypi/emoji/
import emoji

"""
TODO:
- Save and load model in pickle or SQL
- calibrate method to make it check all models and parameters
- In create_model: check for file or calibrate model (be verbose if poss)
- Output as report

"""


class Text_Classifier():
    """
    """
    def __init__(self, clf_text, train_X, train_y):
        """
        """
        # Adjust to a property function based on ending either csv or xlsx
        self.tfidf = TfidfVectorizer(analyzer=self.text_process,
                                     ngram_range=(1, 1))
        self.model = self.create_model(train_X, train_y)

        # Pandas Series or None
        self.clf_text = clf_text
        # Pandas series or None
        self.prediction = None
        # Pandas series or None
        self.proba = None
        # Pandas Dataframe of clf_text & prediction or None
        self.paired = None

    def create_model(self, X, y):
        """
        Returns a fully trained model ready to be used.
        TODO:
        - Integrate calibrate
        - Check if there is a file that can be loaded
        """
        pipe = Pipeline([
            ('vect', self.tfidf),
            ('clf', LogisticRegression(C=10.0, penalty='l2'))
        ])
        pipe.fit(X, y)
        return pipe

    def text_process(self, mess):
        """
        Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords
        3. Returns a list of the cleaned text
        """
        if type(mess) is not str:
            mess = str(mess)
        
        # Check characters to see if they are in punctuation
        nopunc = [char for char in mess if char not in string.punctuation + "–"]

        # Check if char is an emoji
        noemoji = [char for char in nopunc if char not in emoji.UNICODE_EMOJI.keys()]

        # Join the characters again to form the string.
        nopunc = ''.join(noemoji)

        # Now just remove any stopwords
        no_stop = [word for word in nopunc.split() if word.lower() not in
                   stopwords.words('english') if 'http' not in word.lower()]
        
        # Stemming
        snowball = SnowballStemmer('english')
        
        return [snowball.stem(word) for word in no_stop]

    def calibrate(self, fname, X, y):
        """
        Take calibration data and calibrate
        Give choice, with pre-existing values set or completely from scratch (later)
        Use Gridsearch for it and several classifiers for it.
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        pass

    def predict(self, pref_prob=1):
        """
        pref_prob - index of preferred probability of classification
        """
        self.prediction = self.model.predict(self.clf_text)
        self.proba = self.model.predict_proba(self.clf_text)
        self.paired = pd.DataFrame({'text': self.clf_text,
                                    'interesting': self.prediction,
                                    'probability': self.proba[:,pref_prob]})


if __name__ == '__main__':
    clf = pd.read_excel('example_tweets.xlsx')
    df = pd.read_excel('training_data.xlsx')
    ml = Text_Classifier(clf['tweet'], df['tweet'], df['interesting'])
    ml.predict()
    print(ml.paired)
