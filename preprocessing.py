"""Module containing preprocessing function"""
from typing import List
import string

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def lower_each(words_list):
    lower_list = []
    for word in words_list:
        lower_list.append(word.lower())
    return lower_list


def remove_punctuation(words_list):
    clean = []
    for word in words_list:
        clean_word=word.translate(str.maketrans('', '', string.punctuation))
        if clean_word:
            clean.append(clean_word)
    return clean


def remove_stopwords(words: List = None):
    stop_words = set(stopwords.words('english'))
    clean = []
    for word in words:
        if word not in stop_words:
            clean.append(word)
    return clean


def stem_each(words: List = None):
    ps = PorterStemmer()
    return [ps.stem(word) for word in words]


def preprocess_text(text: str = None) -> List:

    tokenized = word_tokenize(text)
    lower = lower_each(tokenized)
    no_punct = remove_punctuation(lower)
    no_stop = remove_stopwords(no_punct)
    stemmed = stem_each(no_stop)

    return stemmed
