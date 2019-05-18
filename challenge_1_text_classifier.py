"""
challenge_1_text_classifier
"""
# 1. Load libraries
from sklearn.datasets import fetch_20newsgroups

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from pdf_reader import convert_pdf_to_txt
import pandas as pd
import os

from nltk.corpus import stopwords

# read pdf files into dict
# each dict record has name of file: list of words, category
# the dict only stores data but can be enriched with another features

# 2. Load data
# classification dataframe
documents_df = pd.read_excel('DTSE_Hackathon_C1_Training/categorization.xlsx')

# Get the list of all files in directory tree at given path
all_files = []
for (dirpath, dirnames, filenames) in os.walk('DTSE_Hackathon_C1_Training'):
    all_files += [os.path.join(dirpath, file) for file in filenames]

all_dfs = []


# Convert all pdf files to txt file
for file in all_files[8:]:
    try:
        converted_file = convert_pdf_to_txt(file)
        base = os.path.basename(file)
        name = os.path.splitext(base)[0]
        print(name)
        file_df = pd.DataFrame({'raw_text': converted_file,
                                'filename': [name]})
        all_dfs.append(file_df)
    except Exception as e:
        raise e

train_df = pd.concat(all_dfs, axis=0, ignore_index=True)
train_df.shape()
train_df.info()


# some dummy
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# TODO: split dataset into training and testing

# 20 test, rest train
# test: 10 FR, 6 JO, 1 misc, 3 PR

# load one file
filo = convert_pdf_to_txt('DTSE_Hackathon_C1_Training/Financial reports/FR100.pdf')

# TODO: add all files into one corpus, so each word/file is compared to the others

# 2b - try to incorporate pictures/charts into dataset
# for example: how many pictures
# one picture at the first page

# 3. preprocess and clean the data

# convert to lowercase

# remove stop words
stop_words = stopwords.words('english')
train_df['without_stopwords'] = [word for word in tokenized_words if word not in stop_words]

# stem/lemmatize the words


# 4. create features for model and clean the data

# cout each word's occurence in document
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape


# discount raw word frequency with tf idf create TF-IDF matrix for each word
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# 4. run ML algorithms
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)

# a) Naive Bayes

# b)



# different approach
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.Consumer_complaint_narrative).toarray()
labels = df.category_id
features.shape