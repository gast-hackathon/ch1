"""
challenge_1_text_classifier
code template used: https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
"""
# 1. import libraries
from pdf_reader import convert_pdf_to_txt
from typing import List
import pandas as pd
import os
import string
import time
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn import naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# 2. Load data
# classification dataframe
categorization_df = pd.read_excel('categorization.xlsx')

# Get the list of all files in directory tree at given path
all_files = []
for (dirpath, dirnames, filenames) in os.walk('DTSE_Hackathon_C1_Training'):
    all_files += [os.path.join(dirpath, file) for file in filenames]


# 2a) Training data
test_files = []
for (dirpath, dirnames, filenames) in os.walk('DTSE_Hackathon_C1_Test'):
    test_files += [os.path.join(dirpath, file) for file in filenames]


all_files.extend(test_files)
test_dfs = []

for file in test_files:
    try:
        converted_file = convert_pdf_to_txt(file)
        name = os.path.basename(file)
        print(name)
        file_df = pd.DataFrame({'raw_text': converted_file,
                                'filename': [name]})
        test_dfs.append(file_df)
    except Exception as e:
        raise e



test_df = pd.concat(test_dfs, axis=0, ignore_index=True)
test_df.info()
test_df.to_csv('ch1_intermediate_test.csv', sep='\t', encoding='utf-8', index=False)


# Convert all pdf files to txt file
start = time.time()
train_dfs = []
for file in all_files:
    try:
        converted_file = convert_pdf_to_txt(file)
        base = os.path.basename(file)
        name = os.path.splitext(base)[0]
        print(name)
        file_df = pd.DataFrame({'raw_text': converted_file,
                                'filename': [name]})
        train_dfs.append(file_df)
    except Exception as e:
        raise e
end = time.time()
end - start

len(train_dfs)
train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
train_df.describe()
train_df.info()

# save the dataframe to csv file for further usage
train_df.to_csv('ch1_intermediate_df.csv', sep='\t', encoding='utf-8', index=False)
train_df = pd.read_csv('ch1_intermediate_df.csv', sep='\t', encoding='utf-8')

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


# 2b - try to incorporate pictures/charts into dataset?
# for example: how many pictures
# one picture at the first page

# TODO: other features should be part of the dataset:
# lenght of document (number of pages), number of pictures, number of tables
# Identifying titles

# 3. preprocess and clean the data

# tokenize words with NLTK
train_df_dedup['tokenized'] = train_df_dedup['raw_text'].map(word_tokenize)
test_df['tokenized'] = test_df['raw_text'].map(word_tokenize)

# convert to lowercase
def lower_each_in_list(words_list):
    lower_list = []
    for word in words_list:
        lower_list.append(word.lower())
    return lower_list

train_df_dedup['lower'] = train_df_dedup['tokenized'].map(lower_each_in_list)
test_df['lower'] = test_df['tokenized'].map(lower_each_in_list)

# remove punctuation
def remove_punctuation_in_list(words_list):
    clean_list = []
    for word in words_list:
        clean_word=word.translate(str.maketrans('', '', string.punctuation))
        if clean_word:
            clean_list.append(clean_word)
    return clean_list


train_df_dedup['nopunc'] = train_df_dedup['lower'].map(remove_punctuation_in_list)
test_df['nopunc'] = test_df['lower'].map(remove_punctuation_in_list)

# remove stop words and punctuation
def remove_stopwords_in_list(words: List = None):
    stop_words = set(stopwords.words('english'))
    clean = []
    for word in words:
        if word not in stop_words:
            clean.append(word)
    return clean

train_df_dedup['without_stopwords'] = train_df_dedup['nopunc'].map(remove_stopwords_in_list)
test_df['without_stopwords'] = test_df['nopunc'].map(remove_stopwords_in_list)

# stem/lemmatize the words

def stem_each_in_list(words: List = None):
    ps = PorterStemmer()
    return [ps.stem(word) for word in words]

train_df_dedup['stemmed'] = train_df_dedup['without_stopwords'].map(stem_each_in_list)
test_df['stemmed'] = test_df['without_stopwords'].map(stem_each_in_list)

train_df_dedup.to_csv('ch1_normalized.csv', sep='\t', encoding='utf-8', index=False)
test_df.to_csv('ch1_normalized_test.csv', sep='\t', encoding='utf-8', index=False)

# train_df_dedup = pd.read_csv('ch1_normalized.csv', sep='\t', encoding='utf-8')
# test_df = pd.read_csv('ch1_normalized_test.csv', sep='\t', encoding='utf-8')

# 4. create features for model and clean the data

# cout each word's occurence in document
count_vect = CountVectorizer(analyzer=None)

# count vectorizer expects text of strings (comma separated words) instead of list of words
train_df_dedup['stemmed_text'] = ["".join(word) for word in train_df_dedup['stemmed'].values]
test_df['stemmed_text'] = ["".join(word) for word in test_df['stemmed'].values]

# split dataset into train and validation based on prepared categorization

labeled_df = pd.merge(train_df_dedup, categorization_df, on='filename')
labeled_df.info

sum(labeled_df['label'] == 'Validation')
train_set = labeled_df[labeled_df['label'] == 'Train']
train_set.info()
validation_set = labeled_df[labeled_df['label'] == 'Validation']
validation_set.info()


# test_set
test_category_df = pd.read_excel('manual_prediction.xlsx')
test_set = pd.merge(test_df, test_category_df, on='filename')
test_set.info()
test_set[['category', 'filename']]

# save files ready for modeling
test_set.to_csv('ch1_test_ready.csv', sep='\t', encoding='utf-8', index=False)
labeled_df.to_csv('ch1_train_ready.csv', sep='\t', encoding='utf-8', index=False)

# 5. prepare NLP features

# a) vector counts
# X_train_counts = count_vect.fit_transform(twenty_train.data)
start = time.time()
count_vect = CountVectorizer(max_features=1500)
count_vect.fit(labeled_df['stemmed_text'])
end = time.time()
end - start
# X_all_counts = count_vect.fit(train_df_dedup['stemmed_text'])
# X_all_counts_t = count_vect.fit_transform(train_df_dedup['stemmed_text'])
# X_all_counts_t.shape

xtrain_count = count_vect.transform(labeled_df['stemmed_text'])
train_category = labeled_df['category']
xtrain_count.shape

# xvalidation_count = count_vect.transform(validation_set['stemmed_text'])
# xvalidation_count.shape

xtest_count = count_vect.transform(test_df['stemmed_text'])
test_category = test_set['category']
xtest_count.shape

# b) TF-IDF -> not completely implemented

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=20000)
tfidf_vect.fit(train_df_dedup['stemmed_text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)


import time
# 6 a) create universal model training function
def train_model(classifier, feature_vector_train, label, feature_vector_test,
                result_test_y, is_neural_net=False, save_model=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    if save_model:
        import pickle
        f = open('MNB_classifier.pickle', 'wb')
        pickle.dump(classifier, f)
        f.close()

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_test)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return {'accuracy': metrics.accuracy_score(predictions, result_test_y),
            'predictions': predictions}

# b) Naive Bayes
start = time.time()
result = train_model(classifier=naive_bayes.MultinomialNB(),
                     feature_vector_train=xtrain_count,
                     label=train_category,
                     feature_vector_test=xtest_count,
                     result_test_y=test_category,
                     save_model=True)
end = time.time()
end - start
print('NB accuracy on word frequency is: ', result['accuracy'])
result_table = pd.DataFrame({'filename': test_set['filename'],
                             'stemmed_text': test_set['stemmed_text'],
                             'predictions': result['predictions'],
                             'category': test_set['category'],
                             'stemmed': test_set['stemmed']})
result_table

result_table[['filename', 'predictions']].sort_values(by='filename')


# Conclusion: No additional models needed as the first one works
# saving results and utility files to use between users
train_sub = labeled_df[['filename', 'category', 'stemmed_text', 'stemmed']]
test_sub = result_table[['filename', 'category', 'stemmed_text', 'stemmed']]

train_test_data = pd.concat([train_sub, test_sub], ignore_index=True,
                            axis=0)
len(train_test_data)

train_test_data.to_csv('train_test_stemmed.csv', sep='\t', encoding='utf-8', index=False)



final_texts = train_test_data['stemmed_text'].tolist()
len(final_texts)
type(final_texts)
final_categories = train_test_data['category'].tolist()
len(final_categories)
rough_filenames = train_test_data['filename'].tolist()
len(rough_filenames)



final_filenames = []
for name in rough_filenames:
    if '.pdf' not in name:
        name += '.pdf'
    final_filenames.append(name)

final_filenames

test_train_tuple = (final_texts, final_categories, final_filenames)
type(test_train_tuple)
type(final_texts)

with open('train_test_tuple.pickle', 'wb') as handle:
    pickle.dump(test_train_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('train_test_tuple.pickle', 'rb') as handle:
    pickle_read = pickle.load(handle)

import ast
type(pickle_read[0])
are_you_list = ast.literal_eval(pickle_read[0][0])
type(are_you_list)
