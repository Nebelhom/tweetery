#!usr/bin/python

import os
import os.path as osp
from pathlib import Path
import pickle
import string

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
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


class Text_Classifier(object):
    """
    """
    def __init__(self, clf_text, train_X=None, train_y=None, ignore_saved=False):
        """
        self._dest = path to data directory
        self._clf_path = path to pickled classifier
        self._hyperparams = path to pre-saved hyperparams
        """
        # Paths
        self._dest = osp.join(os.getcwd(), 'data')
        self._clf_path = osp.join(self._dest, 'classifier.pkl')
        self._hyperparams = osp.join(self._dest, 'hyperparams.pkl')

        # Adjust to a property function based on ending either csv or xlsx
        self.tfidf = TfidfVectorizer(analyzer=self.text_process,
                                     ngram_range=(1, 1))

        self.clf = self.create_classifier(train_X, train_y,
                                          ignore_saved_model=ignore_saved)

        # Pandas Series or None
        self.clf_text = clf_text
        # Pandas series or None
        self.prediction = None
        # Pandas series or None
        self.proba = None
        # Pandas Dataframe of clf_text & prediction or None
        self.paired = None

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
        nopunc = [char for char in mess
                  if char not in string.punctuation + "–"]

        # Check if char is an emoji
        noemoji = [char for char in nopunc
                   if char not in emoji.UNICODE_EMOJI.keys()]

        # Join the characters again to form the string.
        nopunc = ''.join(noemoji)

        # Now just remove any stopwords
        no_stop = [word for word in nopunc.split() if word.lower() not in
                   stopwords.words('english') if 'http' not in word.lower()]

        # Stemming
        snowball = SnowballStemmer('english')

        return [snowball.stem(word) for word in no_stop]

    def create_classifier(self, X=None, y=None, ignore_saved_model=False):
        """
        Returns a fully trained model ready to be used.
        TODO:
        - Integrate calibrate
        - Check if there is a file that can be loaded
        """
        if Path(self._clf_path).is_file() and not ignore_saved_model:
            model = pickle.load(open(self._clf_path, 'rb'))
            print('Classifier successfully loaded')
            return model

        else:
            print('No classifier pre-saved. Please wait while a new ',
                  'classifier is being calibrated...')

            if X is not None and y is not None:
                clf = self.calibrate(X, y)
                print('Classifier has been calibrated.')
                return clf

            else:
                print('You have not defined any training data.',
                      ' Please define:\n')
                if X is None:
                    print('X - pandas Dataframe defining data features.\n')
                if y is None:
                    print('y - pandas Data Series defining the expected',
                          ' classification.\n')
                return None

    def calibrate(self, X, y, full_calibration=False):
        """
        Take calibration data and calibrate
        Give choice, with pre-existing values set or completely from scratch
         (later).
        Use Gridsearch and several classifiers (if possible) for it.
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33,
                                                            random_state=42)
        tfidf = TfidfVectorizer(analyzer=self.text_process)
        pipe = Pipeline([
            ('vect', tfidf),
            ('clf', LogisticRegression())
        ])

        clf = None

        if full_calibration:
            param_grid = [
                {'vect__ngram_range': [(1, 1), (2, 2), (1, 2)],
                 'vect__use_idf': [False, True],
                 'clf__penalty': ['l1', 'l2'],
                 'clf__C': [0.1, 1.0, 10.0, 100.0]
                 }
            ]
            grid = GridSearchCV(pipe, param_grid, scoring='accuracy',
                                cv=10, verbose=10, n_jobs=-1)
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_

        else:
            param_grid = pickle.load(open(self._hyperparams, 'rb'))
            pipe.set_params(**param_grid)
            pipe.fit(X_train, y_train)
            clf = pipe

        return clf

    def predict(self, pref_prob=1):
        """
        pref_prob - index of preferred probability of classification
        """
        self.prediction = self.clf.predict(self.clf_text)
        self.proba = self.clf.predict_proba(self.clf_text)
        self.paired = pd.DataFrame({'text': self.clf_text,
                                    'interesting': self.prediction,
                                    'probability': self.proba[:, pref_prob]})

    def save_classifier(self):
        if self.clf is None:
            print('Cannot save NoneType. Are you sure a classifier is loaded?')
        else:
            if not osp.exists(self._dest):
                os.makedirs(self._dest)
            # Classifier saved
            pickle.dump(self.clf, open(self._clf_path, 'wb'),
                        protocol=4)
            print('Classifier saved.')

            # Hyperparameters saved
            pickle.dump(self.clf.get_params(), open(self._hyperparams, 'wb'),
                        protocol=4)
            print('Hyperparameters of Classifier saved.')


if __name__ == '__main__':
    clf = pd.read_excel('example_tweets.xlsx')
    df = pd.read_excel('training_data.xlsx')
    ml = Text_Classifier(clf['tweet'], df['tweet'], df['interesting'], ignore_saved=True)
    ml.save_classifier()
    ml.predict()
    print(ml.paired)
