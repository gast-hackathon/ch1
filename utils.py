

from os import listdir
from os.path import isfile, join

import pdf_reader
import preprocessing
import ast

from gensim import corpora, models
from gensim.models import Phrases
from collections import defaultdict, OrderedDict

def get_files_in_dir(dirpath):
	return [join(dirpath, f) for f in listdir(dirpath) if isfile(join(dirpath, f))]

def read_files_in_dir(dirpath):
	texts = []

	for file in get_files_in_dir(dirpath):
		txt = pdf_reader.convert_pdf_to_txt(file)

		tokens = preprocessing.preprocess_text(txt)
		texts.append(tokens)

	return texts

def apply_bigrams(texts):
	bigram = Phrases(texts)

	return bigram[texts]

def extract_top_topics(dataset):
	frequency = defaultdict(int)
	for text in dataset:
		for token in text:
			frequency[token] += 1


	# Only keep words that appear more than once
	processed_corpus = [[token for token in text if frequency[token] > 1] for text in dataset]

	dictionary = corpora.Dictionary(processed_corpus)

	# Vectorize dataset
	vec_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

	num_topics = 50
	topics = []
	scores = {}
	lda_model = models.LdaModel(vec_corpus,
								num_topics=num_topics,
								id2word=dictionary,
								passes=10,
								alpha='auto',#[0.01]*num_topics,
								eta='auto', #[0.01]*len(dictionary.keys())
								random_state=1
							)
	lda_model.save("lda.h5")
	for i,topic in lda_model.show_topics(formatted=False, num_topics=num_topics, num_words=1):
		topics.append(topic[0][0])
		scores[topic[0][0]] = 0

	#%% Get score

	for corp in vec_corpus:
		tpcs = lda_model[corp]

		for tp in tpcs:
			topic_name = topics[tp[0]]
			scores[topic_name] += tp[1]

	top_keys = []

	for key, value in sorted(scores.items(), key=lambda item: item[1]):
		top_keys.append(key)

	top_keys.reverse()
	top_topics = []

	for i in range(10):
		key = top_keys[i]
		top_topics.append(key)

		print(key, scores[key])
		
	return lda_model, dictionary, top_topics, topics

def to_list_of_list_of_strings(texts):

	lst = []

	for text in texts.tolist():

		lst2 = []

		for word in ast.literal_eval(text):
			lst2.append(word)
		
		lst.append(lst2)

	return lst
