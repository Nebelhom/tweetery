#!usr/bin/python

import datetime
import os
import os.path as osp

rptname = 'Report-{}'.format(datetime.today())

env_args = {
	'feedfile': osp.join('user_data', 'feed.csv'),
	'keyfile': osp.join('user_data', 'keys.csv'),
    'reporttxt': osp.join('reports', '{}.txt'.format(rptname)),
    'reportdoc': osp.join('reports', '{}.docx'.format(rptname)),
    'reporteml': osp.join('reports', '{}.eml'.format(rptname)),
    'reporttxt': osp.join('reports', '{}.txt'.format(rptname)),
	'train': 'training_data.xlsx'
}