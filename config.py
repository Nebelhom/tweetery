#!usr/bin/python

import datetime
import os.path as osp

rptname = 'Report-{}'.format(datetime.date.today())
tweetname = 'Tweets-{}'.format(datetime.date.today())

env_args = {
        'feedfile': osp.join('user_data', 'example_feeds.csv'),
        'keyfile': osp.join('user_data', 'keys.csv'),
        'reporttxt': osp.join('reports', '{}.txt'.format(rptname)),
        'reportdoc': osp.join('reports', '{}.docx'.format(rptname)),
        'reporteml': osp.join('reports', '{}.eml'.format(rptname)),
        'tweetcsv': osp.join('tweets', '{}.csv'.format(tweetname)),
        'tweetxls': osp.join('tweets', '{}.xlsx'.format(tweetname)),
        'trainfile': 'training_data.xlsx',
}