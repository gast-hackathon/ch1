"""
challenge_1_text_classifier
"""
# 1. Load libraries
from sklearn.model_selection import train_test_split
from pdf_reader import convert_pdf_to_txt
import pandas as pd
import os
import string
from typing import List

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# read pdf files into dict
# each dict record has name of file: list of words, category
# the dict only stores data but can be enriched with another features

# 2. Load data
# classification dataframe
categorization_df = pd.read_excel('categorization.xlsx')

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

len(all_dfs)
train_df = pd.concat(all_dfs, axis=0, ignore_index=True)
train_df.describe()
train_df.info()

# save the dataframe to csv file for further usage
train_df.to_csv('ch1_intermediate_df.csv', sep='\t', encoding='utf-8', index=False)

isok = pd.read_csv('ch1_intermediate_df.csv', sep='\t', encoding='utf-8')

sum(isok['raw_text'] == train_df['raw_text'])


# are there some duplicates? remove them if yes
train_df.iloc[99]['raw_text']
for i in range(len(train_df)):
    if train_df.iloc[i]['raw_text'] == train_df.iloc[100]['raw_text']:
        print(train_df.iloc[i]['filename'])

train_df['raw_text'].duplicated()

# drop duplicates
train_df_dedup = train_df.copy()
train_df_dedup = train_df_dedup.drop_duplicates(subset='raw_text', keep='first')
len(train_df_dedup)

# Done: split dataset into training and testing in categorization.xlsx
# 20 test, rest train
# test: 10 FR, 6 JO, 1 misc, 3 PR


# TODO: add all files into one corpus, so each word/file is compared to the others
# 2b - try to incorporate pictures/charts into dataset
# for example: how many pictures
# one picture at the first page

# 3. preprocess and clean the data

# tokenize words with NLTK
train_df_dedup['tokenized'] = train_df_dedup['raw_text'].map(word_tokenize)

# convert to lowercase
def lower_each_in_list(words_list):
    lower_list = []
    for word in words_list:
        lower_list.append(word.lower())
    return lower_list

train_df_dedup['lower'] = train_df_dedup['tokenized'].map(lower_each_in_list)

# remove punctuation
def remove_punctuation_in_list(words_list):
    clean_list = []
    for word in words_list:
        clean_word=word.translate(str.maketrans('', '', string.punctuation))
        if clean_word:
            clean_list.append(clean_word)
    return clean_list


train_df_dedup['nopunc'] = train_df_dedup['lower'].map(remove_punctuation_in_list)


# remove stop words and punctuation
def remove_stopwords_in_list(words: List = None):
    stop_words = set(stopwords.words('english'))
    clean = []
    for word in words:
        if word not in stop_words:
            clean.append(word)
    return clean

train_df_dedup['without_stopwords'] = train_df_dedup['nopunc'].map(remove_stopwords_in_list)

# stem/lemmatize the words

def stem_each_in_list(words: List = None):
    ps = PorterStemmer()
    return [ps.stem(word) for word in words]

train_df_dedup['stemmed'] = train_df_dedup['without_stopwords'].map(stem_each_in_list)

train_df_dedup.to_csv('ch1_normalized.csv', sep='\t', encoding='utf-8', index=False)



# 4. create features for model and clean the data

# cout each word's occurence in document
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer=None)

# count vectorizer expects text of strings (comma separated words) instead of list of words
train_df_dedup['stemmed_text'] = [" ".join(word) for word in train_df_dedup['stemmed'].values]

# split dataset into train and validation based on prepared categorization

labeled_df = pd.merge(train_df_dedup, categorization_df, on='filename')
labeled_df.info

sum(labeled_df['label'] == 'Validation')
train_set = labeled_df[labeled_df['label'] == 'Train']
train_set.info()
validation_set = labeled_df[labeled_df['label'] == 'Validation']
validation_set.info()


# X_train_counts = count_vect.fit_transform(twenty_train.data)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit(train_df_dedup['stemmed_text'])
X_train_counts_t = count_vect.fit_transform(train_df_dedup['stemmed_text'])

X_train_counts_t.shape

xtrain_count =  count_vect.transform(train_x)


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