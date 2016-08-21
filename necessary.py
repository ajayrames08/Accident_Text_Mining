__author__ = 'AJ'
import pandas as pd
import json
import nltk
import os
#import numpy as np
import pickle
#from sklearn.metrics.pairwise import cosine_similarity
#import multiprocessing as mp
from collections import Counter
import string
from nltk.corpus import stopwords
from nltk import word_tokenize,pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import stem
import collections
global collections
#global operator
import operator
global create_tag_image
global make_tags
global LAYOUTS
#from pytagcloud import create_tag_image, make_tags, LAYOUTS
global get_tag_counts
#import json as simplejson
#from pytagcloud.lang.counter import get_tag_counts
import random
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import pylab as pl
stpwords_extra=["victim","co-worker","worker","employee","employees","die","dead","death","accident","injured"];
causes=["Caught in/between Objects","Collapse of object","Drowning","Electrocution","Exposure to Chemical Substances","Exposure to extreme temperatures","Falls","Fires and Explosion","Other","Struck By Moving Objects","Suffocation"]
