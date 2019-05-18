
#%% Import libs
import importlib
import nltk
import time

from gensim import corpora, models
from gensim.models import Phrases
from collections import defaultdict, OrderedDict

import numpy as np

import pyLDAvis
import pyLDAvis.gensim


#%% Import locals
import pdf_reader
importlib.reload(pdf_reader)
import preprocessing
importlib.reload(preprocessing)
import utils
importlib.reload(utils)


#%% Load dataset

start = time.time()

dataset = utils.read_files_in_dir("C:/Users/Haath/Desktop/DTSE_Hackathon_C2")

print('load dataset', round(time.time() - start), 'sec')
print('samples', len(dataset))


#%% Get topics

topics = utils.extract_top_topics(dataset)

print(topics)
