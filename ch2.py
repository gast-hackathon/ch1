
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

model, dictionary, topics, all_topics = utils.extract_top_topics(dataset)

print(topics)


#%% Cross-reference topics

import pickle
importlib.reload(utils)

threshold = 0.3

print("| Topic | Filename | Category | Relevance |")
print("| ----- | -------- | -------- | --------- |")

texts = [["enlo", "mate"], ["how", "are", "you", "5g"], "I would very much like a wearable".split()]
categories = ["Greetings", "Smt about 5g", "Useless"]
files = ["file1", "file2", "file3"]

f = open("chal1.pickle", "rb")
(texts, categories, files) = pickle.load(f)
f.close()

texts = utils.to_list_of_list_of_strings(texts)

for i in range(len(texts)):
	text = texts[i]

	for topic in model[dictionary.doc2bow(text)]:
		id = topic[0]
		prob = topic[1]
		topic_name = all_topics[id]

		if topic_name in topics and prob > threshold:
			print("| %s | %s | %s | %s %% |" % (all_topics[id], files[i], categories[i], round(topic[1] * 100)))
